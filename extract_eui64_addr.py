"""
   Extract EUI-64 addresses from active IPv6 seed addresses, 
   and generate two EUI-64 seed address files:
1. EUI-64 seed file for the EUI64Gen algorithm: S_eui64_id_top100.csv, 
   containing the complete 32-digit hexadecimal characters (without colons), 
   annotated with network card manufacturer labels.
2. EUI-64 seed file for other algorithms: S_eui64_top100.txt, 
   containing the complete 32-digit hexadecimal characters (with colons), 
   without manufacturer labels.
"""


import csv, os, argparse
from IPy import IP
from tqdm import tqdm

FIRT_COMPANY_ID = 18
FILE_SEED_IPv6 = 'data/seed_ipv6.txt'
FILE_OUI = 'data/oui.txt'



def get_oui_company(file_oui):
    ''' Read the correspondence between OUI and company from oui.txt. '''
    
    # Read oui file.
    with open(file_oui, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    num = len(lines)
    
    # Analyse and process OUI records.
    oui_company = dict()
    for line in tqdm(lines, desc='Analysing oui'):
        if line[11:20] == '(base 16)':
            oui = line[:6]
            company = line[22:].strip()
            oui_company[oui] = company
    return oui_company



def extract_eui64(file_oui, file_seed_ipv6, top_oui=100):
    ''' 
    Extract EUI-64 addresses from IPv6 seed addresses
        Parameter descriptions:
    file_oui: oui.txt, downloadable from https://standards-oui.ieee.org/oui/oui.txt
    file_seed_ipv6: IPv6 seed address file
    top_oui: only extract EUI-64 addresses from the top n manufacturers with the highest number of addresses
    '''
    
    # Read the records from the oui file into the dictionary variable oui_company, in the format oui:company.
    oui_company = get_oui_company(file_oui)
    
    # Read IPv6 seeds file.
    with open(file_seed_ipv6, 'r') as f:
        lines = f.readlines()
    
    # Temporary dictionary variable for the analysis process.
    company_id = dict()
    company_num = dict()
    eui64_company = dict()
    
    # Extract EUI-64 addresses line by line.
    for line in tqdm(lines, desc='Collecting EUI-64'):
        # Represent the IPv6 address as a complete 32-digit hexadecimal string
        norm = IP(line.strip()).strFullsize()
        
        # Skip non-EUI-64 addresses
        if norm[27:32] != 'ff:fe': continue
        
        # Check the U/L flag, retain only global addresses, and skip local addresses.
        flag = int(norm[21],16)
        if flag & 2 == 0: continue
        
        # Get OUI string
        flag_str = hex(flag&13)[-1]
        oui_low = norm[20]+flag_str+norm[22:24]+norm[25:27]
        oui = oui_low.upper()
        
        # Query the manufacturer based on IEEE OUI, 
        # and skip addresses for which the manufacturer cannot be found.
        if oui not in oui_company:
            continue
        else:
            company = oui_company[oui]
        
        # Temporarily store the extracted EUI-64 addresses into a dictionary variable.
        eui64_company[norm] = company
        
        # Accumulate the number of EUI-64 addresses for each manufacturer.
        if company not in company_num:
            company_num[company] = 1
        else:
            company_num[company] += 1

    # Sort the number of EUI-64 addresses for all manufacturers in descending order.
    sorted_count = dict(sorted(company_num.items(), key=lambda item: item[1], reverse=True))
    
    # Number all manufacturers according to their rank based on the number of addresses.
    next_id = FIRT_COMPANY_ID
    company_count = []
    for key, values in sorted_count.items():
        company_id[key] = next_id
        record = [next_id, values, key]
        company_count.append(record)
        next_id += 1
    
    # Get the directory of seed file.
    seed_dir = os.path.dirname(file_seed_ipv6)
    
    # Save each manufacturer's ID and address count to the file oui_count.csv.
    file_company_count = os.path.join(seed_dir, 'oui_count.csv')
    with open(file_company_count, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # Write table head.
        writer.writerow(['ot', 'addrs_numb', 'company'])
        # Write each recorder.
        for item in company_count:
            writer.writerow(item)
         
    # Iterate through each extracted EUI-64 address and annotate it with organization tag.
    seed_eui64 = []     # Variable for saving EUI-64 seeds used by other TGAs.
    seed_eui64_id = []  # Variable for saving EUI-64 seeds used by EUI64Gen.
    for key, values in eui64_company.items():
        # Exclude addresses from manufacturers with lower address counts.
        if company_id[values] >= (FIRT_COMPANY_ID+top_oui): continue
        
        seed_eui64.append(key+'\n')
        addr = key.replace(':','')
        addr_id = f'{addr},{company_id[values]}\n'
        seed_eui64_id.append(addr_id)
    
    # Save the EUI-64 seeds for use by other TGAs.
    file_seed_eui64 = os.path.join(seed_dir, f'S_eui64_top{top_oui:03d}.txt')
    with open(file_seed_eui64, 'w', newline='\n') as f:
        f.writelines(seed_eui64)
        
    # Save the EUI-64 seeds for use by EUI64Gen.
    file_seed_eui64_id = os.path.join(seed_dir, f'S_eui64_id_top{top_oui:03d}.csv')
    with open(file_seed_eui64_id, 'w', newline='\n') as f:
        f.writelines(seed_eui64_id)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_ipv6', default=FILE_SEED_IPv6, type=str, required=False, help='IPv6 seed set file')
    args = parser.parse_args()
    
    extract_eui64(FILE_OUI, args.seed_ipv6, 100)