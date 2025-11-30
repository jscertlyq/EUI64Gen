"""
    The MoETransformerDecoder implementation is derived from and modified based on the torch.nn.transformer code
"""

__version__ = 'v2.0'


# mypy: allow-untyped-defs
import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList


__all__ = ['TransformerDecoder', 'TransformerDecoderLayer']

def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]



class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt)
    """

    __constants__ = ['norm']

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 num_layers: int = 6, dropout: float = 0.1, norm: Optional[nn.Module] = None,
                 num_experts: int = 5, top_k: int = 2,
                 batch_first: bool = False, norm_first: bool = False) -> None:
        super().__init__()
        
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                        dropout=dropout, batch_first=batch_first, norm_first=norm_first,
                                        num_experts=num_experts, top_k=top_k,
                                        activation=activation)
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.norm = norm

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: Optional[bool] = None, ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn1.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
        
        aux_loss = 0
        total_usage = torch.zeros(self.num_experts, device=tgt.device)
        for mod in self.layers:
            output, loss, expert_usage = mod(output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_is_causal=tgt_is_causal)
            aux_loss += loss
            total_usage += expert_usage

        if self.norm is not None:
            output = self.norm(output)

        return output, aux_loss, total_usage/self.num_layers


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt)
    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 num_experts: int = 5, top_k: int = 2,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        super().__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.moe_ffn = MoEFFN(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
                              activation=activation, num_experts=num_experts, top_k=top_k,
                              bias=bias, device=device, dtype=dtype)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
                如果设置tgt_is_causal=True，那么就无需输入atten_mask，MultiheadAttention会自动生成倒三角的因果掩码
                如果想自己输入特别的atten_mask，那么就设置tgt_is_causal=False

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt

        if self.norm_first:
            # 1. Self-attention layer 1
            x = x + self._sa1_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            # 2. Self-attention layer 2
            x = x + self._sa2_block(self.norm2(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            # 3. Feed-forward layer
            moe_out, aux_loss, expert_usage = self.moe_ffn(self.norm3(x))
            x = x + moe_out
        else:        
            # 1. Self-attention layer 1
            x = self.norm1(x + self._sa1_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            # 2. Self-attention layer 2
            x = self.norm2(x + self._sa2_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            # 3. Feed-forward layer
            moe_out, aux_loss, expert_usage = self.moe_ffn(x)
            x = self.norm3(x + moe_out)

        return x, aux_loss, expert_usage

    # self-attention block
    def _sa1_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn1(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _sa2_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn2(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            is_causal=is_causal,
                            need_weights=False)[0]
        return self.dropout2(x)


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal



        

class Router(nn.Module):
    '''
    Router – determines which experts the input sequence is assigned to.
    '''
    
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_layer = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        # x.shape = [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Compute routing weights for each token in the sequence.
        router_logits = self.router_layer(x)  # [batch_size, seq_len, num_experts]
        
        # Use Gumbel-Softmax for differentiable top-k selection.
        if self.training:
            probs = F.gumbel_softmax(router_logits, tau=1.0, hard=False, dim=-1)
        else:
            probs = F.softmax(router_logits, dim=-1)
            
        # Select the top-k experts. Output shape is [batch_size, top_k].
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Normalize probabilities
        expert_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Create expert mask
        expert_mask = torch.zeros_like(probs)
        expert_mask.scatter_(-1, topk_indices, expert_weights)

        return expert_weights, expert_mask, topk_indices

        
class FFN(nn.Module):
    '''
    Define feedforward network model.
    '''
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,                 
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
            
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

        
class MoEFFN(nn.Module):
    '''
    Define MoE FFN.
    '''
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.05,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 num_experts: int = 8, top_k=2,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()         
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.router = Router(d_model, num_experts, top_k)
        self.num_experts=num_experts
        self.top_k=top_k
        self.aux_loss_coef = 0.01
        
        # Experts model pool
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert = FFN(d_model=d_model, dim_feedforward=dim_feedforward,
                         dropout=dropout, activation=activation, bias=bias, device=device, dtype=dtype)
            self.experts.append(expert)
            
    def forward(self, x: Tensor):
        # Routing decision
        
        # expert_weights.shape = [batch_size, seq_len, num_experts]
        expert_weights, expert_mask, expert_indices = self.router(x)
        # expert_indices.shape = [batch_size, seq_len, top_k]
        batch_size, seq_len, top_k = expert_indices.shape
        device = expert_indices.device
        
        # Flatten all token positions and copy to top-k dimension.
        batch_idx_full = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(batch_size, seq_len, top_k)
        seq_idx_full = torch.arange(seq_len, device=device).view(1, seq_len, 1).expand(batch_size, seq_len, top_k)
        
        # Flatten to [batch_size*seq_len*top_k]
        flat_batch = batch_idx_full.reshape(-1)        # [batch_size*seq_len*top_k]
        flat_seq = seq_idx_full.reshape(-1)            # [batch_size*seq_len*top_k]
        flat_experts = expert_indices.reshape(-1)      # [batch_size*seq_len*top_k]
        flat_weights = expert_weights.reshape(-1)      # [batch_size*seq_len*top_k]
        
        # Get the global index and expert assignment for each token.
        sorted_indices = torch.argsort(flat_experts)
        sorted_experts = flat_experts[sorted_indices]
        sorted_batchs = flat_batch[sorted_indices]
        sorted_seq = flat_seq[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_tokens = x[sorted_batchs, sorted_seq]
        
        # Find the starting position of each expert.
        unique_experts, counts = sorted_experts.unique(return_counts=True)
        expert_usage = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        expert_usage[unique_experts] = counts
        
        # Filter out experts with usage == 0.
        nonzero_mask = counts > 0
        unique_experts = unique_experts[nonzero_mask]
        counts = counts[nonzero_mask]
        
        tokens_split = torch.split(sorted_tokens, counts.tolist())
        
        # Forward pass through each expert model
        expert_outputs = []
        for i, expert_idx in enumerate(unique_experts):
            expert = self.experts[expert_idx]
            device = next(expert.parameters()).device
            expert_tokens = tokens_split[i].to(device)
            
            with torch.cuda.device(device):
                output = expert(expert_tokens)
                expert_outputs.append(output)
        
        # Aggregate outputs from expert models.
        concat_expert_outputs = torch.cat(expert_outputs, dim=0)  # [total_tokens, d_model]
        weighted_outputs = concat_expert_outputs * sorted_weights.unsqueeze(1)  # [total_tokens, d_model]
        flat_indices = sorted_batchs * seq_len + sorted_seq  # [total_tokens]
        updates = torch.zeros(batch_size * seq_len, self.d_model).to(device)
        updates.scatter_add_(0, flat_indices.unsqueeze(1).expand(-1, self.d_model), weighted_outputs)
        final_output = updates.view(batch_size, seq_len, self.d_model)
        
        aux_loss = self._compute_aux_loss(expert_usage, expert_mask)
        return self.dropout(final_output), aux_loss, expert_usage
        
        
    def _compute_aux_loss(self, expert_usage, expert_mask):
        '''
        Compute the expert model load-balancing auxiliary loss.
        '''
        
        # Expert model utilization.
        usage_rate = expert_usage / expert_usage.sum()
        
        # Routing probabilities
        router_prob = expert_mask.mean(dim=(0, 1))
        
        # Load balancing loss
        aux_loss = self.num_experts * torch.sum(usage_rate * router_prob)
        return aux_loss * self.aux_loss_coef
