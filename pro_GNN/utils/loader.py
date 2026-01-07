# ──────────────────────────
# utils/loader.py
# ──────────────────────────
from encoder.gnn_encoder import GNNEncoder
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles
from torch_geometric.data import Batch
import torch
import deepchem as dc

featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)



def load_encoder(args):
    encoder = GNNEncoder(
        node_in=args.node_in,
        edge_in=args.edge_in,
        hidden_dim=args.hidden_dim,
        out_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    return encoder

def pair_collate_fn(batch):
    data1_list, data2_list, label_list, ref_list, prb_list = zip(*batch)
    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)
    labels = torch.stack(label_list)
    ref_values = torch.stack(ref_list)
    prb_values = torch.stack(prb_list)

    return batch1, batch2, labels, ref_values, prb_values


def smiles_to_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol_feats = featurizer.featurize([mol])[0]

    x = torch.tensor(mol_feats.node_features, dtype=torch.float)
    edge_index = torch.tensor(mol_feats.edge_index, dtype=torch.long)
    edge_attr = torch.tensor(mol_feats.edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)



# # CSVから (data_A, data_B, label) のリストを作成
# def load_pair_dataset_from_csv(csv_path):
#     df = pd.read_csv(csv_path)
#     data_list = []
#     for _, row in df.iterrows():
#         smiles_a = row['REF-SMILES']
#         smiles_b = row['PRB-SMILES']
#         label = row['DELTA-standard_value']
#         try:
#             data_a = smiles_to_data(smiles_a)
#             data_b = smiles_to_data(smiles_b)
#             data_list.append((data_a, data_b, label))
#         except ValueError as e:
#             print(e)
#     return data_list
