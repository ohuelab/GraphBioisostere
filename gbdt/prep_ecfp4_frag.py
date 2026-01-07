from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import joblib
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("target_dir", type=str, default="MMP_dataset/")
args = parser.parse_args()


if os.path.exists(os.path.join(args.target_dir, "dataset_consistentsmiles.joblib")):
    print(f"Dataset already exists in {args.target_dir}")
    exit()

data = pd.read_csv(f"{args.target_dir}/dataset_consistentsmiles.csv", index_col=0)

def smiles_to_mol(smiles_list: List[str]) -> List[Optional[Chem.Mol]]:
    """
    Convert a list of SMILES strings to RDKit molecules.
    Returns a list of molecules (None for invalid SMILES).
    """
    mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)
    return mols

def batch_smiles_to_fp(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
    n_threads: int = 4
) -> np.ndarray:
    """
    Convert a batch of SMILES strings to Morgan fingerprints efficiently.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius
        n_bits: Number of bits in fingerprint
        n_threads: Number of threads to use for computation

    Returns:
        NumPy array of fingerprints (n_valid_mols, n_bits)
    """
    # Convert SMILES to molecules
    mols = [mol for mol in smiles_to_mol(smiles_list) if mol is not None]

    if not mols:
        raise ValueError("No valid molecules found in the input SMILES list")

    # Create fingerprint generator once
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    # Generate fingerprints in parallel
    fps_tuple = fpgen.GetFingerprints(mols, numThreads=n_threads)

    # Convert to numpy array efficiently
    result = np.zeros((len(fps_tuple), n_bits), dtype=np.int8)

    for i, fp in enumerate(fps_tuple):
        # Convert fingerprint directly to the pre-allocated array
        arr_view = result[i]
        DataStructs.ConvertToNumpyArray(fp, arr_view)

    return result

# データの変換関数
def convert_records(df, radius=2, n_bits=2048, n_threads=4):
    records = []
    # Create a dictionary to store unique SMILES and their corresponding fingerprints
    smiles_to_fp_dict = {}

    # First, process all unique SMILES strings
    unique_smiles = list(set(df['REF-FRAG'].tolist() + df['PRB-FRAG'].tolist()))
    # Process all unique SMILES in batches for efficiency
    batch_size = 1000
    for i in range(0, len(unique_smiles), batch_size):
        batch_smiles = unique_smiles[i:i+batch_size]
        try:
            # Get valid molecules
            valid_mols = [(smi, mol) for smi, mol in zip(batch_smiles, smiles_to_mol(batch_smiles)) if mol is not None]
            if not valid_mols:
                continue

            valid_smiles = [smi for smi, _ in valid_mols]
            fps = batch_smiles_to_fp(valid_smiles, radius=radius, n_bits=n_bits, n_threads=n_threads)

            # Store fingerprints in dictionary
            for idx, (smi, _) in enumerate(valid_mols):
                smiles_to_fp_dict[smi] = fps[idx]

        except Exception as e:
            print(f"Error processing batch: {e}")

    # Then create the pairs using the cached fingerprints
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating pairs"):
        try:
            smi1 = row['REF-FRAG']
            smi2 = row['PRB-FRAG']
            label_bin = row['label_bin']
            delta_value = row['delta_value']

            # Use the pre-processed fingerprints
            if smi1 in smiles_to_fp_dict and smi2 in smiles_to_fp_dict:
                fp1 = smiles_to_fp_dict[smi1]
                fp2 = smiles_to_fp_dict[smi2]

                pair = {
                    'fp1': fp1,
                    'fp2': fp2,
                    'label': float(label_bin),
                    'delta_value': float(delta_value),
                    'ref_smiles': smi1,
                    'prb_smiles': smi2
                }
                records.append(pair)
            else:
                print(f"Skipping row {row.name}: SMILES not found in processed dictionary")
        except Exception as e:
            print(f"Error processing row {row.name}: {e}")
    return records

# ECFP4 fingerprints (radius=2) for Morgan fingerprints

os.makedirs(args.target_dir, exist_ok=True)
records = convert_records(data, radius=2, n_bits=2048, n_threads=4)
print(f"Number of records: {len(records)}/ {len(data)}")
joblib.dump(records, os.path.join(args.target_dir, "dataset_consistentsmiles-frag.joblib"))
