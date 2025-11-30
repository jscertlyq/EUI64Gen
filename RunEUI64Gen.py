"""
    Experimental code for EUI64Gen which is an effective EUI-64 target generation algorithm.
"""

import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
import random
import sys
import math
import numpy as np
from datetime import datetime
from MoETransformerDecoder import TransformerDecoder
from ordered_set import OrderedSet


# Parameters related to model training
SEED_FILE = 'data/S_eui64_id_top100.csv'
MODEL_FILE  = 'data/modeleui64gen.pth'
CANDIDATES_FILE = 'data/candidates.txt'
MAX_LEN = 35    # <bos> + <ot> + 32 nibbles + <eos>
BATCH_SIZE = 64
DATA_SHUFFLE = True
EPOCH_NUM = 30
LEARNING_RATE=5e-5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Model hyperparameters
N_LAYER = 6             # the number of Transformer Decoder layers
N_HEAD = 8              # the number of attention heads
D_FORWARD_DIM = 2048    # d_ff
D_MODEL = 512           # d_model
DROPOUT = 0.05          # dropout rate
TEMPERATURE = 0.8       # softmax temperature
NUM_EXPERTS = 5         # Number of expert models
TOP_K = 2               # Number of activated expert models



# Token-related variables
BOS, EOS = '<bos>', '<eos>'
tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', BOS, EOS]
token_to_id = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
               'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, BOS:16, EOS:17}
id_to_token = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
               10:'a', 11:'b', 12:'c', 13:'d', 14:'e', 15:'f', 16:BOS, 17:EOS}
BOS_ID = token_to_id[BOS]
EOS_ID = token_to_id[EOS]
TOKEN_NUM = 18
ORG_NUM = 100           # Organization tab (OT) number
DICT_SIZE = TOKEN_NUM + ORG_NUM


def token_encode(tokens, ot):
    """ 
    Convert tokens to IDs
    nibble list -> <bos>ID + <ot>ID + nibble ID list + <eos>ID 
    """
    token_ids = [BOS_ID, ot]    # Start token ID + <ot> ID
    # Traverse nibble list and convert each token into ID.
    for token in tokens:
        token_ids.append(token_to_id[token])
    token_ids.append(EOS_ID) # End token ID
    return token_ids



def token_decode(token_ids):
    """ 
    Convert IDs to tokens
    <bos>ID + <ot>ID + nibble ID list + <eos>ID -> nibble list
    """
    tokens = []
    for idx in token_ids[2:-1]:
        # Skip start, ot and end tokens
            tokens.append(id_to_token[idx])
    return tokens    


class IPv6AddrSet(Dataset):
    """ Define custom IPv6 address dataset class """
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    

def load_data(seed_file=SEED_FILE, batch_size=BATCH_SIZE):
    """ Load the IPv6 address dataset from seed file and return a DataLoader. """
    with open(seed_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    # Encode the IPv6 address as a 32-length list of integers (0–15), 
    # with <bos> token ID at the beginning, ot at the second place, and <eos> token ID at the end.
    address = []
    for line in raw_data:
        addr, ouid = line.split(',')
        ot = int(ouid)
        address.append(token_encode(addr, ot))
    dataset = IPv6AddrSet(np.array(address))
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=DATA_SHUFFLE)
    return dataloader




class ModelEUI64Gen(nn.Module):
    """
    Define EUI64Gen model
    """
    def __init__(self, dict_size=DICT_SIZE, d_model=D_MODEL, nhead=N_HEAD,
                 d_ff=D_FORWARD_DIM, num_layers=N_LAYER, dropout=DROPOUT, 
                 activation=F.gelu, num_experts=8, top_k=2):
        super(ModelEUI64Gen, self).__init__()
        
        # Dictionary size
        self.dict_size = dict_size
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=d_model)
        
        # Layer normalization layer
        norm = nn.LayerNorm(d_model)
        
        # An N-layer Transformer decoder stack
        self.decoder = TransformerDecoder(d_model=d_model, batch_first=True, nhead=nhead, dropout=dropout,
                                          dim_feedforward=d_ff, num_layers=num_layers,
                                          norm=norm, activation=activation, num_experts=num_experts, top_k=top_k)
        
        # Linear output layer
        self.predictor = nn.Linear(d_model, dict_size)

    def forward(self, tgt, device=DEVICE):
        # Generate self-attention mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)

        # word embedding
        tgt = self.embedding(tgt)

        # permute(1, 0, 2) for reordering the tgt dimension to place the batch in the middle as batch_first is not enabled.
        out, aux_loss, expert_usage = self.decoder(tgt, tgt_mask=tgt_mask) 
        out = self.predictor(out)
        return out, aux_loss, expert_usage




def train_model(model, seed_file=SEED_FILE, model_file=None,
                batch_size=BATCH_SIZE, lr=LEARNING_RATE, epochs=EPOCH_NUM, 
                device=DEVICE):
    """ model training """
     
    dataloader = load_data(seed_file, batch_size) 

    # odel Loss Function and Optimizer 
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        data_progress = tqdm(dataloader, desc="Train...")
        for step, data in enumerate(data_progress, start=1):
            data = data.to(device)

            # Construct the training data and target data
            tgt = data[:, :-1]
            tgt_y = data[:, 1:]
            
            # Perform the Transformer computation
            out, aux_loss, expert_usage = model(tgt, device)
            main_loss = criteria(out.permute(0,2,1).contiguous(), tgt_y.to(dtype=torch.long))    # batch_first = True
            loss = main_loss + aux_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            # Update the training progress
            total_loss += loss.item()
            data_progress.set_description(f"Train... [epoch {epoch}/{epochs}, loss {(total_loss / step):.5f}]")

    # Save model parameters if needed
    if model_file is not None:
        torch.save(model.state_dict(), model_file)
    
    # Return the final average training loss.
    return total_loss / step




def ids_to_ipv6(addr):
    ''' Transform a list of 32 nibbles into an IPv6 address in colon-separated format. '''
    ipv6 = ''
    for i in range(len(addr)):
        ipv6 += id_to_token[addr[i]]
        if i%4 == 3 and i < 31:
            ipv6 += ':'
    return ipv6



def gen_addr_batch(model, temperature, head_num, head_batch, device=DEVICE):
    """ Generate one batch of IPv6 address """
    
    with torch.no_grad():
        # convert IPv6 address head nibble into a tensor.
        head_batch = torch.tensor(head_batch, dtype=torch.long, device=device)
        batch_size = head_batch.shape[0]

        # Strip the last <eos> token.
        tgt = head_batch[:,:-1]

        i = 0
        while i < 32 - head_num:
            # model forward, out.shape=(sequence_len, batch_size, embed_dim)
            out, _, _ = model(tgt, device)

             # 'out' contains the probability distribution over all tokens in the vocabulary. 
             # Only take the first 16 tokens from the dictionary, i.e., 0-9, a-f.
            _probas = out[:, -1, :16]       # batch_first = True
            
            # Softmax temperature scaling.
            _probas = _probas/temperature

            # Replace all values below top_k with -∞.
            indices_to_remove = _probas < torch.topk(_probas, 16)[0][..., -1, None]
            _probas[indices_to_remove] = -float('Inf')

            # Apply the softmax operation so that tokens with higher probabilities are more likely to be selected.
            _probas = F.softmax(_probas, dim=-1)

            # Randomly select one token from the top-k based on their probabilities.
            y = torch.multinomial(_probas, num_samples=1)
            
            # Concatenate the selected token to the previously generated result.
            tgt = torch.cat((tgt, y), dim=-1)
            
            # Check whether the step has reached FFFE; 
            # if so, directly append FFFE without generating from the model.
            if i == 21-head_num:
                fffe = [15, 15, 15, 14]
                fffe_batch = [fffe for i in range(batch_size)]
                fffe_tensor = torch.tensor(fffe_batch, dtype=torch.long, device=device)
                tgt = torch.cat((tgt, fffe_tensor), dim=-1)
                i += 5
            else:
                i += 1

        # Remove <bos> <ot> token and return generated addresses.
        ipv6list = list(map(ids_to_ipv6, tgt[:, 2:].tolist()))
        return ipv6list

# Organization tags with high hit rates discovered through experiments.
HIGH_HIT_OT = [89, 55, 94, 45, 97, 43, 61, 64, 81, 82, 63, 92, 103, 107, 78, 115, 109, 33, 79]        

def generate_target(model, ot, budget, candidate_file, temperature=TEMPERATURE, batch_size=BATCH_SIZE, device=DEVICE, head='2'):
    ''' Generate a certain number (budget) of IPv6 addresses and write them to a file. '''
    
    head_num = len(head)    # the length of address head nibble
    model.eval()            # Switch the model to evaluation mode.

    # Generate IPv6 addresses in batches.
    addrs = OrderedSet()
    progress_bar = tqdm(total=budget, desc="Generating...") # Display a progress bar
    while len(addrs) < budget:
        # If ot==0, it means to randomly generate an OT.
        if ot == 0:
            # otd = random.randint(18, model.dict_size-1)  # Randomly generate an OT.
            otd = random.choice(HIGH_HIT_OT)   # Randomly select high hit rate an OT.
        else:
            otd = ot
    
        # encode address head nibble
        head_tokens_ids = token_encode(head, otd)
    
        # Copy a single address head nibble into batch mode.
        head_batch = [head_tokens_ids for i in range(batch_size)]
        
        # Generate one batch EUI-64 addresses.
        gen_addr = gen_addr_batch(model, temperature, head_num, head_batch, device=device)
        addrs.update(gen_addr)
        
        # Adjust the progress bar based on the number of IPv6 addresses to be created.
        progress_bar.n = len(addrs)
        progress_bar.refresh()
            
    # Append a newline character to the end of each IPv6 address.
    addrn = list(map(lambda s: s + "\n", addrs))
    
    # Write the generated addresses to a file.
    with open(candidate_file, 'w') as f:
        f.writelines(addrn[:budget])




if __name__ == '__main__':
    '''
    Run example:
    python RunEUI64Gen.py --seed_file=data/S_eui64_id_top100.csv \
                          --model_file=data/modeleui64gen.pth \
                          --candidate_file=data/eui64addrs.txt \
                          --batch_size=64 \
                          --epochs=50 \
                          --learning_rate=5e-5 \
                          --temperature=0.8 \
                          --device=cuda:0 \
                          --budget=100000
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_train', action='store_true', default=False, help='no train flag')
    parser.add_argument('--seed_file', default=SEED_FILE, type=str, required=False, help='IPv6 seed set file for training')
    parser.add_argument('--model_file', default=MODEL_FILE, type=str, required=False, help='model parameters file')
    parser.add_argument('--candidate_file', default=CANDIDATES_FILE, type=str, required=False, help='generated candidates file')
    parser.add_argument('--org_num', default=ORG_NUM, type=int, required=False, help='number of organizations included in seeds')
    parser.add_argument('--n_layer', default=N_LAYER, type=int, required=False, help='the number of TransformerDecdoer layers')
    parser.add_argument('--n_head', default=N_HEAD, type=int, required=False, help='the number of self-attention heads')
    parser.add_argument('--d_ff', default=D_FORWARD_DIM, type=int, required=False, help='feed-forward dimension')
    parser.add_argument('--d_model', default=D_MODEL, type=int, required=False, help='model dimension')
    parser.add_argument('--dropout', default=DROPOUT, type=float, required=False, help='dropout rate')
    parser.add_argument('--temperature', default=TEMPERATURE, type=float, required=False, help='softmax temperature')
    parser.add_argument('--num_experts', default=NUM_EXPERTS, type=int, required=False, help='Number of expert models')
    parser.add_argument('--top_k', default=TOP_K, type=int, required=False, help='Number of activated expert models')
    parser.add_argument('--epochs', default=EPOCH_NUM, type=int, required=False, help='training epochs')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, required=False, help='batch size during model training and evaluation')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, required=False, help='learning rate during training') 
    parser.add_argument('--budget', default=1000000, type=int, required=False, help='the number candidate addresses to be generated')
    parser.add_argument('--ot', default=0, type=int, required=False, help='Organization tag，If 0 then randomly generate')
    parser.add_argument('--device', default=DEVICE, type=str, required=False, help='training and evaluation device')
    args = parser.parse_args()

    # Construct the model
    model = ModelEUI64Gen(dict_size=TOKEN_NUM+args.org_num, d_model=args.d_model, nhead=args.n_head, d_ff=args.d_ff, 
                            num_layers=args.n_layer, dropout=args.dropout, num_experts=args.num_experts, top_k=args.top_k).to(args.device)
    
    # Model training
    if args.no_train:
        model.load_state_dict(torch.load(args.model_file), weights_only=True)  # load model parameters
    else:
        train_model(model=model, seed_file=args.seed_file, model_file=args.model_file,
                batch_size=args.batch_size, lr=args.learning_rate, epochs=args.epochs, 
                device=args.device)          

    # Generate candidate address
    generate_target(model=model, ot=args.ot, temperature=args.temperature, budget=args.budget, 
                    candidate_file=args.candidate_file, batch_size=2048, device=args.device)

