#!/usr/bin/env python3
"""
ターゲット別ファインチューニング用データセット作成スクリプト

Butina clusteringベースの5-fold CVでデータを分割し、
各ターゲットのMMPデータをグラフデータに変換する。

Author: Generated for transfer learning experiments
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina

# プロジェクトのルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.loader import smiles_to_data

# ターゲット情報
TARGET_INFO = {
    "BACE": "CHEMBL4822",
    "CDK2": "CHEMBL301", 
    "JNK1": "CHEMBL2276",
    "MCL1": "CHEMBL4361",
    "P38": "CHEMBL260",
    "PTP1B": "CHEMBL335",
    "thrombin": "CHEMBL204",
    "TYK2": "CHEMBL3553"
}


def get_morgan_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048):
    """SMILESからMorgan fingerprintを計算"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)


def get_unique_molecules(df: pd.DataFrame) -> List[str]:
    """データフレームからユニークな分子のリストを取得"""
    ref_mols = set(df["REF-SMILES"].tolist())
    prb_mols = set(df["PRB-SMILES"].tolist())
    return list(ref_mols | prb_mols)


def butina_cluster(smiles_list: List[str], cutoff: float = 0.6) -> List[int]:
    """
    Butina clusteringを実行
    
    Args:
        smiles_list: SMILESのリスト
        cutoff: Tanimoto距離のカットオフ（0.6 = 類似度0.4）
        
    Returns:
        各分子のクラスタIDのリスト
    """
    # フィンガープリントの計算
    fps = []
    valid_indices = []
    for i, smiles in enumerate(smiles_list):
        fp = get_morgan_fingerprint(smiles)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)
    
    if len(fps) < 2:
        return [0] * len(smiles_list)
    
    # 距離行列の計算（1 - Tanimoto similarity）
    n = len(fps)
    dists = []
    for i in range(1, n):
        for j in range(0, i):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists.append(1 - sim)
    
    # Butina clustering
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    
    # クラスタIDの割り当て
    cluster_ids = [-1] * len(smiles_list)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            original_idx = valid_indices[idx]
            cluster_ids[original_idx] = cluster_id
    
    # 無効な分子には-1が入っているので、別クラスタとして扱う
    max_cluster = max(cluster_ids)
    for i, cid in enumerate(cluster_ids):
        if cid == -1:
            max_cluster += 1
            cluster_ids[i] = max_cluster
    
    return cluster_ids


def create_5fold_from_clusters(
    smiles_list: List[str], 
    cluster_ids: List[int],
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    クラスタをサイズ順に5フォールドに分配
    
    Args:
        smiles_list: SMILESのリスト
        cluster_ids: 各分子のクラスタID
        seed: ランダムシード（同じサイズのクラスタをシャッフル）
        
    Returns:
        各フォールドに属する分子インデックスのDict
    """
    np.random.seed(seed)
    
    # クラスタごとの分子インデックスを集計
    cluster_to_mols = defaultdict(list)
    for mol_idx, cluster_id in enumerate(cluster_ids):
        cluster_to_mols[cluster_id].append(mol_idx)
    
    # クラスタをサイズ順にソート（大きい順）
    sorted_clusters = sorted(
        cluster_to_mols.items(), 
        key=lambda x: -len(x[1])
    )
    
    # 5フォールドに分配
    folds = [[] for _ in range(5)]
    fold_sizes = [0] * 5
    
    for cluster_id, mol_indices in sorted_clusters:
        # 最も小さいフォールドに追加
        min_fold = np.argmin(fold_sizes)
        folds[min_fold].extend(mol_indices)
        fold_sizes[min_fold] += len(mol_indices)
    
    # 各フォールド内でシャッフル
    for fold in folds:
        np.random.shuffle(fold)
    
    return {f"cv{i}": fold for i, fold in enumerate(folds)}


def create_pair_dataset(
    df: pd.DataFrame,
    train_indices: List[int],
    test_indices: List[int],
    smiles_to_idx: Dict[str, int]
) -> Tuple[List, List]:
    """
    ペアデータセットを作成
    
    Args:
        df: MMPペアのDataFrame
        train_indices: 訓練用分子のインデックス
        test_indices: テスト用分子のインデックス
        smiles_to_idx: SMILESからインデックスへのマッピング
        
    Returns:
        (train_pairs, test_pairs): 訓練・テストペアのリスト
    """
    train_set = set(train_indices)
    test_set = set(test_indices)
    
    train_pairs = []
    test_pairs = []
    
    for _, row in df.iterrows():
        ref_smiles = row["REF-SMILES"]
        prb_smiles = row["PRB-SMILES"]
        delta_value = row["delta_value"]
        
        ref_idx = smiles_to_idx.get(ref_smiles, -1)
        prb_idx = smiles_to_idx.get(prb_smiles, -1)
        
        if ref_idx == -1 or prb_idx == -1:
            continue
        
        # グラフに変換
        ref_graph = smiles_to_data(ref_smiles)
        prb_graph = smiles_to_data(prb_smiles)
        
        if ref_graph is None or prb_graph is None:
            continue
        
        pair_data = (ref_graph, prb_graph, torch.tensor([delta_value], dtype=torch.float32))
        
        # 両方の分子がテストセットに含まれる場合のみテストデータ
        if ref_idx in test_set and prb_idx in test_set:
            test_pairs.append(pair_data)
        # 両方がtrainセット、またはどちらかがtrainセットの場合は訓練データ
        elif ref_idx in train_set or prb_idx in train_set:
            # 少なくとも一方がテストに含まれていない場合は訓練データ
            if ref_idx not in test_set and prb_idx not in test_set:
                train_pairs.append(pair_data)
        
    return train_pairs, test_pairs


def prepare_target_dataset(
    target_name: str,
    data_dir: Path,
    output_dir: Path,
    seed: int = 42
):
    """
    特定ターゲットのデータセットを準備
    
    Args:
        target_name: ターゲット名（例: "BACE"）
        data_dir: 入力データのディレクトリ
        output_dir: 出力先ディレクトリ
        seed: ランダムシード
    """
    target_id = TARGET_INFO.get(target_name)
    if target_id is None:
        print(f"Unknown target: {target_name}")
        return
    
    print(f"\n=== Processing {target_name} (ID: {target_id}) ===")
    
    # 出力ディレクトリ作成
    target_output_dir = output_dir / target_name
    target_output_dir.mkdir(parents=True, exist_ok=True)
    
    # データ読み込み
    target_df_path = data_dir / target_name / "target_df.csv"
    if not target_df_path.exists():
        print(f"Data file not found: {target_df_path}")
        return
    
    df = pd.read_csv(target_df_path)
    print(f"Loaded {len(df)} pairs")
    
    # ユニーク分子の抽出
    unique_smiles = get_unique_molecules(df)
    print(f"Unique molecules: {len(unique_smiles)}")
    
    # SMILESからインデックスへのマッピング
    smiles_to_idx = {s: i for i, s in enumerate(unique_smiles)}
    
    # Butina clustering
    print("Running Butina clustering...")
    cluster_ids = butina_cluster(unique_smiles, cutoff=0.6)
    n_clusters = len(set(cluster_ids))
    print(f"Number of clusters: {n_clusters}")
    
    # 5-fold CV分割
    folds = create_5fold_from_clusters(unique_smiles, cluster_ids, seed=seed)
    
    # クラスタ情報を保存
    butina_info = {
        "smiles_list": unique_smiles,
        "cluster_ids": cluster_ids,
        "folds": folds,
        "n_clusters": n_clusters
    }
    
    with open(target_output_dir / "butina-5cv.pkl", "wb") as f:
        pickle.dump(butina_info, f)
    
    # 各フォールドのデータセットを作成
    for fold_name, test_indices in folds.items():
        print(f"\nProcessing {fold_name}...")
        
        # テスト以外のインデックスを訓練データとする
        all_indices = set(range(len(unique_smiles)))
        train_indices = list(all_indices - set(test_indices))
        
        # ペアデータセット作成
        train_pairs, test_pairs = create_pair_dataset(
            df, train_indices, test_indices, smiles_to_idx
        )
        
        print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
        
        # 保存
        fold_idx = int(fold_name.replace("cv", ""))
        dataset = {
            "train": train_pairs,
            "test": test_pairs,
            "train_indices": train_indices,
            "test_indices": test_indices
        }
        
        torch.save(dataset, target_output_dir / f"dataset_cv{fold_idx}.pt")
    
    # ターゲットDataFrameもコピー
    df.to_csv(target_output_dir / "target_df.csv", index=False)
    
    print(f"\n✅ {target_name} dataset prepared at {target_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare target datasets for finetuning")
    parser.add_argument("--data_dir", type=str, 
                        default=None,
                        help="Input data directory (default: pro_GNN/notebooks/target)")
    parser.add_argument("--output_dir", type=str,
                        default=None,
                        help="Output directory (default: finetune/target_datasets)")
    parser.add_argument("--targets", type=str, nargs="+",
                        default=["BACE", "JNK1", "P38", "thrombin", "PTP1B", "CDK2"],
                        help="Target names to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # デフォルトパスの設定
    script_dir = Path(__file__).parent
    pro_gnn_dir = script_dir.parent
    bioiso_dir = pro_gnn_dir.parent
    
    if args.data_dir is None:
        data_dir = bioiso_dir / "notebooks" / "target"
    else:
        data_dir = Path(args.data_dir)
    
    if args.output_dir is None:
        output_dir = script_dir / "target_datasets"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Targets: {args.targets}")
    
    for target in args.targets:
        prepare_target_dataset(target, data_dir, output_dir, seed=args.seed)
    
    print("\n" + "="*50)
    print("All datasets prepared!")


if __name__ == "__main__":
    main()
