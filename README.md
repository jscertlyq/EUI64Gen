# EUI64Gen

EUI64Gen is a Transformer decoder-based EUI-64 target generation algorithm. It not only achieves a higher hit rate in EUI-64 address detection compared to current state-of-the-art TGAs, but also has the capability to generate EUI-64 candidates for specific organizations, supporting targeted detection. 

## Runtime environment

* Python 3.9.1
* pytorch 2.5.1
* pytorch-cuda 12.4

## Seed set

[IPv6 Hitlist](https://ipv6hitlist.github.io/) provides an IPv6 Hitlist Service to publish responsive IPv6 addresses, aliased prefixes, and non-aliased prefixes.

## Run example

1. Running the EUI64Gen experimental program with default parameters will train the EUI64Gen model for 30 epochs using the provided demonstration EUI-64 seeds. The trained model will then generate 1 million EUI-64 candidate addresses with randomly assigned manufacturer labels, saving them to the file data/candidates.txt.
If you want to targetively detect EUI-64 addresses of a specific manufacturer, you can look up the corresponding tag for that manufacturer in the data/oui_count.csv file, and adjust the --ot parameter during generation accordingly.
The trained model parameters are saved in data/modeleui64gen.pth. When generating candidate addresses multiple times, you can use the --on_train parameter to skip retraining and directly load the pre-trained model parameters to generate candidates.
```shell
python RunEUI64Gen.py --seed_file=data/S_eui64_id_top100.csv --budget=1000000  --ot=0
```

2. If you have your own IPv6 seed addresses, you can use the extract_eui64_addr.py program to extract EUI-64 seed addresses from the IPv6 addresses. This process will also generate a new manufacturer address and tag file, oui_count.csv. You can then use the extracted EUI-64 seeds to retrain the model.
```shell
python extract_eui64_addr.py --seed_ipv6=data/your_seed_ipv6.txt
```
