
# Usage: python prep_cls_all.py /path/to/MMP_dataset/


from utils.loader import smiles_to_data
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import os
import torch

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("target_dir", type=str, default="MMP_dataset/")
args = parser.parse_args()


if os.path.exists(os.path.join(args.target_dir, "dataset_consistentsmiles.pt")):
    print(f"Dataset already exists in {args.target_dir}")
    exit()

data = pd.read_csv(f"{args.target_dir}/dataset_consistentsmiles.csv", index_col=0)
class MoleculePairDataset(Dataset):
    def __init__(self, pair_list):
        """
        pair_list: [{'data1': Data, 'data2': Data, 'label': Tensor}, ...]
        """
        self.pair_list = pair_list

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        item = self.pair_list[idx]
        return item['data1'], item['data2'], item['label']

# データの変換関数
def convert_records(df):
    records = []
    # Create a dictionary to store unique SMILES and their corresponding data objects
    smiles_to_data_dict = {}

    # First, process all unique SMILES strings
    unique_smiles = set(df['REF-SMILES'].tolist() + df['PRB-SMILES'].tolist())
    for smi in tqdm(unique_smiles, desc="Processing unique SMILES"):
        try:
            smiles_to_data_dict[smi] = smiles_to_data(smi)
            smiles_to_data_dict[smi].smiles = smi  # SMILES を Data に埋め込む
        except Exception as e:
            print(f"Error processing SMILES {smi}: {e}")

    # Then create the pairs using the cached data objects
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating pairs"):
        try:
            smi1 = row['REF-SMILES']
            smi2 = row['PRB-SMILES']
            label_bin = row['label_bin']
            delta_value = row['delta_value']

            # Use the pre-processed data objects
            if smi1 in smiles_to_data_dict and smi2 in smiles_to_data_dict:
                data1 = smiles_to_data_dict[smi1]
                data2 = smiles_to_data_dict[smi2]

                pair = {
                    'data1': data1,
                    'data2': data2,
                    'label': torch.tensor([label_bin], dtype=torch.float),
                    'delta_value': torch.tensor([delta_value], dtype=torch.float),
                }
                records.append(pair)
            else:
                print(f"Skipping row {row.name}: SMILES not found in processed dictionary")
        except Exception as e:
            print(f"Error processing row {row.name}: {e}")
    return records

records = convert_records(data)

X = MoleculePairDataset(records)
os.makedirs(args.target_dir, exist_ok=True)


torch.save(X,
    os.path.join(args.target_dir, "dataset_consistentsmiles.pt")
)
