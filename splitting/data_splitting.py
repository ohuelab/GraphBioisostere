#!/usr/bin/env python3
"""
Integrated Data Splitting Script

Read CSV file and perform 5-fold splitting based on TID (target ID),
with TID frequency filtering and leak removal across different targets.

Features:
- Maintain target-based 5-fold CV
- Test data filtering by TID frequency (optional)
  - Only MMPs appearing in a specified number of TIDs or more are included in test
  - Enables evaluation of prediction accuracy for more general transformation patterns
- Remove MMP leaks across different targets (train only, optional)
  - full_test: Remove from train based on entire original test set
  - filtered_test: Remove from train based only on filtered test (improves train data retention)
- Remove fragment leaks across different targets (train only, optional)
- Output detailed statistics for each processing step
- Generate new pkl and .pt files

Processing Flow:
1. TID-based 5-fold splitting
2. TID frequency filtering (optional) - test data only
3. Leak removal (optional) - train data only
   - leak_removal_mode="full_test": Leak removal based on entire original test
   - leak_removal_mode="filtered_test": Leak removal based only on filtered test

Optimization:
Using leak_removal_mode="filtered_test" improves train data retention rate

Usage examples:
    # Basic splitting only (no filtering)
    python data_splitting.py \
        --csv_path ../MMP_dataset/dataset_consistentsmiles.csv \
        --output_dir ../pro_GNN/dataset/dataset_consistentsmiles_41 \
        --data_path ../MMP_dataset/dataset_consistentsmiles.pt \
        --pkl_output tid_5cv_consistentsmiles_41.pkl \
        --seed 41
    
    # MMP leak removal only
    python data_splitting.py \
        --csv_path ../MMP_dataset/dataset_consistentsmiles.csv \
        --output_dir ../pro_GNN/dataset/dataset_consistentsmiles_41_no_leak \
        --data_path ../MMP_dataset/dataset_consistentsmiles.pt \
        --pkl_output tid_5cv_consistentsmiles_41_no_leak.pkl \
        --seed 41 --remove_mmp_leak
    
    # Molecule leak removal (prevents 3-way leaks, recommended)
    python data_splitting.py \
        --csv_path ../MMP_dataset/dataset_consistentsmiles.csv \
        --output_dir ../pro_GNN/dataset/dataset_consistentsmiles_molecule \
        --data_path ../MMP_dataset/dataset_consistentsmiles.pt \
        --pkl_output tid_5cv_consistentsmiles_molecule.pkl \
        --remove_molecule_leak \
    
    # TID3+ filtering + molecule leak removal (most strict)
    python data_splitting.py \
        --csv_path ../MMP_dataset/dataset_consistentsmiles.csv \
        --output_dir ../pro_GNN/dataset/dataset_consistentsmiles_tid3_molecule \
        --data_path ../MMP_dataset/dataset_consistentsmiles.pt \
        --pkl_output tid_5cv_consistentsmiles_tid3_molecule.pkl \
        --min_tid_count 3 --remove_molecule_leak \
        --leak_removal_mode filtered_test
"""

import pandas as pd
import numpy as np
import pickle
import torch
import random
import argparse
import os
import json
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from datetime import datetime
from torch.utils.data import Dataset


class MoleculePairDataset(Dataset):
    """
    分子ペアデータセット
    """
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


def verify_input_files(csv_path: str, pt_path: str) -> bool:
    """
    入力ファイルの整合性を事前確認
    
    Args:
        csv_path: CSVファイルのパス
        pt_path: .ptファイルのパス
    
    Returns:
        検証が成功した場合True、失敗した場合False
    """
    print("\n" + "="*70)
    print("Verifying input files alignment...")
    print("="*70)
    
    try:
        # Check CSV file length
        df = pd.read_csv(csv_path)
        csv_length = len(df)
        print(f"CSV file length: {csv_length}")
        
        # Check .pt file length
        pt_data = torch.load(pt_path, weights_only=False)
        pt_length = len(pt_data)
        print(f".pt file length: {pt_length}")
        
        # Verify length match
        if csv_length != pt_length:
            print(f"\n✗ ERROR: Length mismatch!")
            print(f"  CSV has {csv_length} records")
            print(f"  .pt has {pt_length} records")
            print(f"  Difference: {abs(csv_length - pt_length)}")
            print("\nPlease ensure CSV and .pt files are aligned before proceeding.")
            return False
        else:
            print(f"\n✓ Lengths match: {csv_length} records")
            print("Input files verification passed!")
            return True
    
    except Exception as e:
        print(f"\n✗ ERROR during verification: {str(e)}")
        return False


def load_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load CSV file and return each row as a dict.
    Required columns:
      REF-SMILES, PRB-SMILES, REF-standard_value, PRB-standard_value, TID
    """
    print(f"\nLoading CSV file: {path}")
    
    df = pd.read_csv(path)
    
    # Add log transformation
    df["DELTA-log10"] = df["REF-standard_value"].apply(np.log10) - df["PRB-standard_value"].apply(np.log10)
    df["REF-log10"] = df["REF-standard_value"].apply(np.log10)
    df["PRB-log10"] = df["PRB-standard_value"].apply(np.log10)
    
    # Standardize column names
    df = df.rename(columns={
        "REF-SMILES": "ref_smiles",
        "PRB-SMILES": "prb_smiles",
        "TID": "tid"
    })
    
    df["label"] = df["DELTA-log10"].astype(float)
    df["ref_value"] = df["REF-log10"].astype(float)
    df["prb_value"] = df["PRB-log10"].astype(float)
    df["index"] = range(len(df))
    
    # Select only necessary columns and convert to list of dicts
    columns = ["ref_smiles", "prb_smiles", "label", "ref_value", "prb_value", "index", "tid"]
    result = df[columns].to_dict('records')
    
    print(f"Loaded {len(result)} records from CSV")
    print(f"Unique TIDs: {len(df['tid'].unique())}")
    
    return result


def get_mmp_key(ref_smiles: str, prb_smiles: str) -> str:
    """
    MMPペアのユニークなキーを生成（順序を正規化）
    """
    return tuple(sorted([ref_smiles, prb_smiles]))


def get_frag_key(ref_smiles: str, prb_smiles: str) -> str:
    """
    フラグメントペアのユニークなキーを生成（順序を正規化）
    """
    return tuple(sorted([ref_smiles, prb_smiles]))


def collect_mmp_tid_frequency(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Set[str]]:
    """
    データセット全体でMMPごとのTID出現頻度を計算
    
    Args:
        records: 全データレコード
    
    Returns:
        MMPキー -> TIDのセットのマッピング
    """
    print(f"\n{'='*60}")
    print("Collecting MMP-TID frequency across entire dataset...")
    print(f"{'='*60}")
    
    mmp_to_tids = defaultdict(set)
    
    for record in records:
        mmp_key = get_mmp_key(record["ref_smiles"], record["prb_smiles"])
        mmp_to_tids[mmp_key].add(record["tid"])
    
    print(f"Found {len(mmp_to_tids)} unique MMPs")
    
    # Display TID frequency distribution
    tid_counts = [len(tids) for tids in mmp_to_tids.values()]
    print(f"\nTID frequency distribution:")
    print(f"  Min: {min(tid_counts)}")
    print(f"  Max: {max(tid_counts)}")
    print(f"  Mean: {np.mean(tid_counts):.2f}")
    print(f"  Median: {np.median(tid_counts):.0f}")
    
    # Display number of MMPs per frequency
    from collections import Counter
    freq_dist = Counter(tid_counts)
    print(f"\nMMPs by TID frequency:")
    for freq in sorted(freq_dist.keys())[:10]:  # Show top 10
        print(f"  TID={freq}: {freq_dist[freq]} MMPs ({freq_dist[freq]/len(mmp_to_tids)*100:.2f}%)")
    if len(freq_dist) > 10:
        print(f"  ... (and {len(freq_dist)-10} more frequencies)")
    
    return dict(mmp_to_tids)


def filter_test_by_tid_frequency(
    cv_splits: List[Tuple],
    mmp_to_tids: Dict[Tuple[str, str], Set[str]],
    min_tid_count: int,
    verbose: bool = True
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    Filter test data by TID frequency
    Keep only MMPs that appear in the specified number of TIDs or more
    
    Args:
        cv_splits: CV split results
        mmp_to_tids: Mapping from MMP key to set of TIDs
        min_tid_count: Minimum TID occurrence count
        verbose: Display detailed output
    
    Returns:
        Filtered CV split results and statistics
    """
    print(f"\n{'='*60}")
    print(f"Filtering test data by TID frequency (min_tid_count={min_tid_count})...")
    print("Train and validation data remain unchanged")
    print(f"{'='*60}")
    
    stats = {
        "min_tid_count": min_tid_count,
        "total_removed": 0,
        "per_fold": []
    }
    
    filtered_splits = []
    
    for fold_idx, (train, val, test) in enumerate(cv_splits):
        # Filter test data
        test_filtered = []
        removed_count = 0
        
        for record in test:
            mmp_key = get_mmp_key(record["ref_smiles"], record["prb_smiles"])
            tid_count = len(mmp_to_tids.get(mmp_key, set()))
            
            if tid_count >= min_tid_count:
                test_filtered.append(record)
            else:
                removed_count += 1
        
        filtered_splits.append((train, val, test_filtered))
        
        fold_stats = {
            "fold": fold_idx,
            "original_test": len(test),
            "filtered_test": len(test_filtered),
            "removed": removed_count,
            "retention_rate": len(test_filtered) / len(test) * 100 if len(test) > 0 else 0
        }
        stats["per_fold"].append(fold_stats)
        stats["total_removed"] += removed_count
        
        if verbose:
            print(f"Fold {fold_idx}: Kept {len(test_filtered)}/{len(test)} "
                  f"({fold_stats['retention_rate']:.2f}%) test samples")
    
    total_original = sum(fold["original_test"] for fold in stats["per_fold"])
    total_filtered = sum(fold["filtered_test"] for fold in stats["per_fold"])
    overall_retention = total_filtered / total_original * 100 if total_original > 0 else 0
    
    print(f"\nOverall: Kept {total_filtered}/{total_original} "
          f"({overall_retention:.2f}%) test samples")
    print(f"Total removed: {stats['total_removed']}")
    
    return filtered_splits, stats


def split_by_tid_fv(
    records: List[Dict[str, Any]],
    n_folds: int,
    seed: int
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    TIDに基づいてn-fold cross validationの分割を行う
    
    Args:
        records: データレコードのリスト
        n_folds: フォールド数
        seed: 乱数シード
    
    Returns:
        各フォールドの(train, validation, test)タプルのリスト
    """
    if not any("tid" in r for r in records):
        raise ValueError("No 'tid' column in records")
    
    print(f"\n{'='*60}")
    print(f"Performing {n_folds}-fold split with seed={seed}")
    print(f"{'='*60}")
    
    rng = random.Random(seed)
    tid2idx = defaultdict(list)
    
    # Collect indices together
    for i, r in enumerate(records):
        tid2idx[r["tid"]].append(i)
    
    all_tids = list(tid2idx.keys())
    rng.shuffle(all_tids)
    
    print(f"Found {len(all_tids)} unique TIDs")
    
    # Split TIDs into n groups for n-fold CV
    fold_size = len(all_tids) // n_folds
    tid_folds = [all_tids[i:i + fold_size] for i in range(0, len(all_tids), fold_size)]
    
    # Adjust if last fold is too small
    if len(tid_folds) > n_folds:
        tid_folds[-2].extend(tid_folds[-1])
        tid_folds.pop()
    
    results = []
    
    # Use each fold as test set
    for test_fold_idx in range(n_folds):
        # Use next fold as validation set
        val_fold_idx = (test_fold_idx + 1) % n_folds
        
        # Set TIDs for test and validation
        test_tids = set(tid_folds[test_fold_idx])
        val_tids = set(tid_folds[val_fold_idx])
        
        tr_idx, va_idx, te_idx = [], [], []
        
        # Assign indices for each TID
        for tid in all_tids:
            idxs = tid2idx[tid]
            if tid in test_tids:
                te_idx.extend(idxs)
            elif tid in val_tids:
                va_idx.extend(idxs)
            else:
                tr_idx.extend(idxs)
        
        # Get records using indices at once
        fold_result = (
            [records[i] for i in tr_idx],
            [records[i] for i in va_idx],
            [records[i] for i in te_idx]
        )
        
        results.append(fold_result)
        
        print(f"Fold {test_fold_idx}: train={len(fold_result[0])}, val={len(fold_result[1])}, test={len(fold_result[2])}")
    
    return results


def remove_mmp_leaks(
    cv_splits: List[Tuple],
    original_splits: List[Tuple] = None,
    use_filtered_test: bool = False,
    verbose: bool = True
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    異なるターゲット間のMMPリーク（同一分子ペア）を除外する
    testデータに出現するペアをtrainデータから除外する（validationは保持）
    
    Args:
        cv_splits: CV分割結果（フィルタリング後の可能性あり）
        original_splits: 元のCV分割結果（use_filtered_test=Falseの場合に使用）
        use_filtered_test: Trueの場合cv_splitsのtestを使用、Falseの場合original_splitsのtestを使用
        verbose: 詳細な出力を表示
    
    Returns:
        リーク除外後のCV分割結果と統計情報
    """
    print(f"\n{'='*60}")
    print("Removing MMP leaks (identical molecule pairs)...")
    print("Removing from train only (validation is kept)")
    if use_filtered_test:
        print("Using FILTERED test data for leak removal (preserves more train data)")
    else:
        print("Using FULL ORIGINAL test data for leak removal")
    print(f"{'='*60}")
    
    stats = {
        "total_removed": 0,
        "per_fold": [],
        "mode": "filtered_test" if use_filtered_test else "full_test"
    }
    
    # リーク除外の基準となるtestデータを決定
    reference_splits = cv_splits if use_filtered_test else original_splits
    if reference_splits is None:
        reference_splits = cv_splits  # フォールバック
    
    cleaned_splits = []
    
    for fold_idx, ((train, val, test), (ref_train, ref_val, ref_test)) in enumerate(zip(cv_splits, reference_splits)):
        # 基準となるTestのMMPセットを作成
        test_mmps = set()
        for record in ref_test:
            mmp_key = get_mmp_key(record["ref_smiles"], record["prb_smiles"])
            test_mmps.add(mmp_key)
        
        # Trainからのみリークを除外（Valは保持）
        train_cleaned = []
        removed_count_train = 0
        
        for record in train:
            mmp_key = get_mmp_key(record["ref_smiles"], record["prb_smiles"])
            if mmp_key not in test_mmps:
                train_cleaned.append(record)
            else:
                removed_count_train += 1
        
        removed_count = removed_count_train
        cleaned_splits.append((train_cleaned, val, test))
        
        fold_stats = {
            "fold": fold_idx,
            "original_train": len(train),
            "original_val": len(val),
            "cleaned_train": len(train_cleaned),
            "cleaned_val": len(val),
            "removed_train": removed_count_train,
            "removed_val": 0,
            "removed_total": removed_count,
            "removal_rate_train": removed_count_train / len(train) * 100 if len(train) > 0 else 0,
            "removal_rate_val": 0.0
        }
        stats["per_fold"].append(fold_stats)
        stats["total_removed"] += removed_count
        
        if verbose:
            print(f"Fold {fold_idx}: Removed {removed_count_train}/{len(train)} "
                  f"({fold_stats['removal_rate_train']:.2f}%) from train")
    
    print(f"\nTotal MMP leaks removed: {stats['total_removed']}")
    
    return cleaned_splits, stats


def remove_frag_leaks(
    cv_splits: List[Tuple],
    original_splits: List[Tuple] = None,
    use_filtered_test: bool = False,
    verbose: bool = True
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    異なるターゲット間のフラグメントリーク（同一フラグメントペア）を除外する
    testデータに出現するフラグメントペア（順不同）をtrainデータから除外する（validationは保持）
    
    Args:
        cv_splits: CV分割結果（フィルタリング後の可能性あり）
        original_splits: 元のCV分割結果（use_filtered_test=Falseの場合に使用）
        use_filtered_test: Trueの場合cv_splitsのtestを使用、Falseの場合original_splitsのtestを使用
        verbose: 詳細な出力を表示
    
    Returns:
        リーク除外後のCV分割結果と統計情報
    """
    print(f"\n{'='*60}")
    print("Removing fragment leaks (identical fragment pairs)...")
    print("Removing from train only (validation is kept)")
    if use_filtered_test:
        print("Using FILTERED test data for leak removal (preserves more train data)")
    else:
        print("Using FULL ORIGINAL test data for leak removal")
    print(f"{'='*60}")
    
    stats = {
        "total_removed": 0,
        "per_fold": [],
        "mode": "filtered_test" if use_filtered_test else "full_test"
    }
    
    # リーク除外の基準となるtestデータを決定
    reference_splits = cv_splits if use_filtered_test else original_splits
    if reference_splits is None:
        reference_splits = cv_splits  # フォールバック
    
    cleaned_splits = []
    
    for fold_idx, ((train, val, test), (ref_train, ref_val, ref_test)) in enumerate(zip(cv_splits, reference_splits)):
        # 基準となるTestのフラグメントペアセットを作成（順不同）
        test_frags = set()
        for record in ref_test:
            frag_key = get_frag_key(record["ref_smiles"], record["prb_smiles"])
            test_frags.add(frag_key)
        
        # Trainからのみリークを除外（Valは保持）
        train_cleaned = []
        removed_count_train = 0
        
        for record in train:
            frag_key = get_frag_key(record["ref_smiles"], record["prb_smiles"])
            if frag_key not in test_frags:
                train_cleaned.append(record)
            else:
                removed_count_train += 1
        
        removed_count = removed_count_train
        cleaned_splits.append((train_cleaned, val, test))
        
        fold_stats = {
            "fold": fold_idx,
            "original_train": len(train),
            "original_val": len(val),
            "cleaned_train": len(train_cleaned),
            "cleaned_val": len(val),
            "removed_train": removed_count_train,
            "removed_val": 0,
            "removed_total": removed_count,
            "removal_rate_train": removed_count_train / len(train) * 100 if len(train) > 0 else 0,
            "removal_rate_val": 0.0
        }
        stats["per_fold"].append(fold_stats)
        stats["total_removed"] += removed_count
        
        if verbose:
            print(f"Fold {fold_idx}: Removed {removed_count_train}/{len(train)} "
                  f"({fold_stats['removal_rate_train']:.2f}%) from train")
    
    print(f"\nTotal fragment leaks removed: {stats['total_removed']}")
    
    return cleaned_splits, stats


def remove_molecule_leaks(
    cv_splits: List[Tuple],
    original_splits: List[Tuple] = None,
    use_filtered_test: bool = False,
    verbose: bool = True
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    「3すくみリーク」を防ぐために、testデータに含まれる全分子をtrainデータから除外する
    A-B間とB-C間のラベルからA-C間のラベルが推測できてしまう問題に対処
    
    Args:
        cv_splits: CV分割結果（フィルタリング後の可能性あり）
        original_splits: 元のCV分割結果（use_filtered_test=Falseの場合に使用）
        use_filtered_test: Trueの場合cv_splitsのtestを使用、Falseの場合original_splitsのtestを使用
        verbose: 詳細な出力を表示
    
    Returns:
        リーク除外後のCV分割結果と統計情報
    """
    print(f"\n{'='*60}")
    print("Removing molecule-level leaks (3-way transitivity leak)...")
    print("Removing train samples that contain ANY molecule from test data")
    print("Removing from train only (validation is kept)")
    if use_filtered_test:
        print("Using FILTERED test data for leak removal (preserves more train data)")
    else:
        print("Using FULL ORIGINAL test data for leak removal")
    print(f"{'='*60}")
    
    stats = {
        "total_removed": 0,
        "per_fold": [],
        "mode": "filtered_test" if use_filtered_test else "full_test"
    }
    
    # リーク除外の基準となるtestデータを決定
    reference_splits = cv_splits if use_filtered_test else original_splits
    if reference_splits is None:
        reference_splits = cv_splits  # フォールバック
    
    cleaned_splits = []
    
    for fold_idx, ((train, val, test), (ref_train, ref_val, ref_test)) in enumerate(zip(cv_splits, reference_splits)):
        # 基準となるTestに含まれる全分子のセットを作成
        test_molecules = set()
        for record in ref_test:
            test_molecules.add(record["ref_smiles"])
            test_molecules.add(record["prb_smiles"])
        
        if verbose:
            print(f"Fold {fold_idx}: Test set contains {len(test_molecules)} unique molecules")
        
        # Trainから、testに含まれる分子を持つペアを除外（Valは保持）
        train_cleaned = []
        removed_count_train = 0
        
        for record in train:
            # trainのペアがtestの分子を含んでいるかチェック
            if record["ref_smiles"] in test_molecules or record["prb_smiles"] in test_molecules:
                removed_count_train += 1
            else:
                train_cleaned.append(record)
        
        removed_count = removed_count_train
        cleaned_splits.append((train_cleaned, val, test))
        
        fold_stats = {
            "fold": fold_idx,
            "original_train": len(train),
            "original_val": len(val),
            "cleaned_train": len(train_cleaned),
            "cleaned_val": len(val),
            "removed_train": removed_count_train,
            "removed_val": 0,
            "removed_total": removed_count,
            "removal_rate_train": removed_count_train / len(train) * 100 if len(train) > 0 else 0,
            "removal_rate_val": 0.0,
            "test_molecules": len(test_molecules)
        }
        stats["per_fold"].append(fold_stats)
        stats["total_removed"] += removed_count
        
        if verbose:
            print(f"Fold {fold_idx}: Removed {removed_count_train}/{len(train)} "
                  f"({fold_stats['removal_rate_train']:.2f}%) from train")
    
    print(f"\nTotal molecule-level leaks removed: {stats['total_removed']}")
    
    return cleaned_splits, stats


def save_statistics(
    output_dir: str,
    initial_splits: List[Tuple],
    final_splits: List[Tuple],
    mmp_stats: Dict[str, Any] = None,
    frag_stats: Dict[str, Any] = None,
    tid_filter_stats: Dict[str, Any] = None,
    molecule_stats: Dict[str, Any] = None
):
    """
    除外前後の統計情報を保存
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    stats_file = os.path.join(output_dir, "split_statistics.json")
    text_file = os.path.join(output_dir, "split_statistics.txt")
    
    # 統計情報を収集
    statistics = {
        "timestamp": datetime.now().isoformat(),
        "initial_statistics": [],
        "final_statistics": [],
        "tid_frequency_filter": tid_filter_stats,
        "mmp_leak_removal": mmp_stats,
        "fragment_leak_removal": frag_stats,
        "molecule_leak_removal": molecule_stats
    }
    
    for i, (train, val, test) in enumerate(initial_splits):
        statistics["initial_statistics"].append({
            "fold": i,
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(train) + len(val) + len(test)
        })
    
    for i, (train, val, test) in enumerate(final_splits):
        statistics["final_statistics"].append({
            "fold": i,
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(train) + len(val) + len(test)
        })
    
    # JSON形式で保存
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    print(f"\nStatistics saved to: {stats_file}")
    
    # テキスト形式でも保存
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("Data Split Statistics\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {statistics['timestamp']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("Initial Split Statistics\n")
        f.write("-"*70 + "\n")
        for stat in statistics["initial_statistics"]:
            f.write(f"Fold {stat['fold']}: train={stat['train']}, "
                   f"val={stat['val']}, test={stat['test']}, "
                   f"total={stat['total']}\n")
        
        if tid_filter_stats:
            f.write("\n" + "-"*70 + "\n")
            f.write(f"TID Frequency Filtering (min_tid_count={tid_filter_stats['min_tid_count']})\n")
            f.write("-"*70 + "\n")
            f.write(f"Total removed from test: {tid_filter_stats['total_removed']}\n")
            for fold_stat in tid_filter_stats["per_fold"]:
                f.write(f"Fold {fold_stat['fold']}: "
                       f"test: {fold_stat['filtered_test']}/{fold_stat['original_test']} "
                       f"({fold_stat['retention_rate']:.2f}%)\n")
        
        if mmp_stats:
            f.write("\n" + "-"*70 + "\n")
            f.write("MMP Leak Removal (from train only)\n")
            f.write(f"Mode: {mmp_stats.get('mode', 'N/A')}\n")
            f.write("-"*70 + "\n")
            f.write(f"Total removed: {mmp_stats['total_removed']}\n")
            for fold_stat in mmp_stats["per_fold"]:
                f.write(f"Fold {fold_stat['fold']}: "
                       f"train: {fold_stat['removed_train']}/{fold_stat['original_train']} "
                       f"({fold_stat['removal_rate_train']:.2f}%)\n")
        
        if frag_stats:
            f.write("\n" + "-"*70 + "\n")
            f.write("Fragment Leak Removal (from train only)\n")
            f.write(f"Mode: {frag_stats.get('mode', 'N/A')}\n")
            f.write("-"*70 + "\n")
            f.write(f"Total removed: {frag_stats['total_removed']}\n")
            for fold_stat in frag_stats["per_fold"]:
                f.write(f"Fold {fold_stat['fold']}: "
                       f"train: {fold_stat['removed_train']}/{fold_stat['original_train']} "
                       f"({fold_stat['removal_rate_train']:.2f}%)\n")
        
        if molecule_stats:
            f.write("\n" + "-"*70 + "\n")
            f.write("Molecule-Level Leak Removal (3-way transitivity, from train only)\n")
            f.write(f"Mode: {molecule_stats.get('mode', 'N/A')}\n")
            f.write("-"*70 + "\n")
            f.write(f"Total removed: {molecule_stats['total_removed']}\n")
            for fold_stat in molecule_stats["per_fold"]:
                f.write(f"Fold {fold_stat['fold']}: "
                       f"train: {fold_stat['removed_train']}/{fold_stat['original_train']} "
                       f"({fold_stat['removal_rate_train']:.2f}%), "
                       f"test molecules: {fold_stat.get('test_molecules', 'N/A')}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("Final Split Statistics\n")
        f.write("-"*70 + "\n")
        for stat in statistics["final_statistics"]:
            f.write(f"Fold {stat['fold']}: train={stat['train']}, "
                   f"val={stat['val']}, test={stat['test']}, "
                   f"total={stat['total']}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"Statistics saved to: {text_file}")


def save_cv_splits_as_pkl(cv_splits: List[Tuple], output_path: str):
    """
    CV分割結果をpklファイルとして保存
    """
    print(f"\nSaving CV splits to: {output_path}")
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(cv_splits, f)
    
    print(f"Successfully saved pkl file")


def create_dataset_pt_files(cv_splits: List[Tuple], data_path: str, output_dir: str):
    """
    CV分割結果とtorchデータを使用してdataset_cv{i}.ptファイルを作成
    
    Args:
        cv_splits: CV分割結果
        data_path: 元のtorch データセットファイルのパス
        output_dir: 出力ディレクトリ
    """
    print(f"\nLoading torch dataset from: {data_path}")
    data = torch.load(data_path, weights_only=False)
    print(f"Loaded torch dataset with {len(data)} items")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(len(cv_splits)):
        print(f"Creating dataset_cv{i}.pt...")
        
        tr, va, te = cv_splits[i]
        
        # インデックスを取得
        tr_index = [a["index"] for a in tr]
        va_index = [a["index"] for a in va]
        te_index = [a["index"] for a in te]
        
        # データを取得
        tr_data = [data[j] for j in tr_index]
        va_data = [data[j] for j in va_index]
        te_data = [data[j] for j in te_index]
        
        # 保存
        output_path = os.path.join(output_dir, f"dataset_cv{i}.pt")
        torch.save({"train": tr_data, "valid": va_data, "test": te_data}, output_path)
        
        print(f"Saved {output_path}: train={len(tr_data)}, valid={len(va_data)}, test={len(te_data)}")
    
    print(f"\nAll dataset files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="CSV to torch dataset splitting with TID filtering and leak removal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic split without any filtering
  python data_splitting.py --csv_path data.csv --data_path data.pt \\
    --output_dir output/ --pkl_output split.pkl
  
  # With MMP leak removal only
  python data_splitting.py --csv_path data.csv --data_path data.pt \\
    --output_dir output/ --pkl_output split.pkl --remove_mmp_leak
  
  # With TID3+ filtering and leak removal (recommended)
  python data_splitting.py --csv_path data.csv --data_path data.pt \\
    --output_dir output/ --pkl_output split.pkl --min_tid_count 3 --remove_mmp_leak
        """
    )
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to CSV file")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to torch dataset.pt file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for dataset_cv{i}.pt files")
    parser.add_argument("--pkl_output", type=str, required=True,
                       help="Output path for pkl file")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=40,
                       help="Random seed (default: 40)")
    parser.add_argument("--min_tid_count", type=int, default=None,
                       help="Minimum TID count for test data filtering (e.g., 3 for TID3+). "
                            "If not specified, no TID filtering is applied.")
    parser.add_argument("--remove_mmp_leak", action="store_true",
                       help="Remove MMP leaks (identical molecule pairs from train)")
    parser.add_argument("--remove_frag_leak", action="store_true",
                       help="Remove fragment leaks (identical fragment pairs from train)")
    parser.add_argument("--remove_molecule_leak", action="store_true",
                       help="Remove molecule-level leaks (3-way transitivity leak): "
                            "removes all train samples containing ANY molecule that appears in test data. "
                            "This prevents A-B and B-C pairs from leaking information about A-C pairs.")
    parser.add_argument("--leak_removal_mode", type=str, default="full_test",
                       choices=["full_test", "filtered_test"],
                       help="Leak removal mode: 'full_test' uses original full test data for removal, "
                            "'filtered_test' uses TID-filtered test data only (preserves more train data). "
                            "Only relevant when both --min_tid_count and leak removal options are specified. "
                            "(default: full_test)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Data Splitting with TID Filtering and Leak Removal")
    print("="*70)
    print(f"CSV path: {args.csv_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"PKL output: {args.pkl_output}")
    print(f"N-folds: {args.n_folds}")
    print(f"Seed: {args.seed}")
    print(f"Min TID count: {args.min_tid_count if args.min_tid_count else 'None (no filtering)'}")
    print(f"Remove MMP leak: {args.remove_mmp_leak}")
    print(f"Remove fragment leak: {args.remove_frag_leak}")
    print(f"Remove molecule leak: {args.remove_molecule_leak}")
    if args.remove_mmp_leak or args.remove_frag_leak or args.remove_molecule_leak:
        print(f"Leak removal mode: {args.leak_removal_mode}")
    print("="*70)
    
    # 入力ファイルの整合性を検証
    if not verify_input_files(args.csv_path, args.data_path):
        print("\n" + "="*70)
        print("ABORTED: Input files verification failed!")
        print("="*70)
        exit(1)
    
    # CSV ファイルを読み込み
    records = load_csv(args.csv_path)
    
    # TID ベースで分割
    initial_splits = split_by_tid_fv(records, args.n_folds, args.seed)
    
    # 処理フロー: TIDフィルタリング → リーク除外
    # TIDフィルタリングを先に行うことで、リーク除外の対象を減らし、
    # trainデータの保持率を向上させる（leak_removal_mode="filtered_test"の場合）
    current_splits = initial_splits
    original_splits = initial_splits  # リーク除外の基準として保存
    tid_filter_stats = None
    mmp_stats = None
    frag_stats = None
    
    # Step 1: TID頻度によるテストデータフィルタリング（オプション）
    if args.min_tid_count is not None:
        print(f"\n{'='*70}")
        print(f"Step 1: TID Frequency Filtering")
        print(f"{'='*70}")
        
        # データセット全体でMMP-TID頻度を計算
        mmp_to_tids = collect_mmp_tid_frequency(records)
        
        # テストデータをフィルタリング
        current_splits, tid_filter_stats = filter_test_by_tid_frequency(
            current_splits,
            mmp_to_tids,
            args.min_tid_count
        )
    
    # Step 2: リーク除外（オプション）
    molecule_stats = None
    if args.remove_mmp_leak or args.remove_frag_leak or args.remove_molecule_leak:
        print(f"\n{'='*70}")
        print(f"Step 2: Leak Removal")
        print(f"{'='*70}")
        
        # リーク除外モードの決定
        use_filtered_test = (args.leak_removal_mode == "filtered_test" and args.min_tid_count is not None)
        
        if use_filtered_test:
            print("Using leak_removal_mode='filtered_test': Removing leaks based on FILTERED test data")
            print("This preserves more training data compared to 'full_test' mode")
        else:
            print("Using leak_removal_mode='full_test': Removing leaks based on FULL ORIGINAL test data")
        
        if args.remove_mmp_leak:
            current_splits, mmp_stats = remove_mmp_leaks(
                current_splits,
                original_splits=original_splits,
                use_filtered_test=use_filtered_test
            )
        
        if args.remove_frag_leak:
            current_splits, frag_stats = remove_frag_leaks(
                current_splits,
                original_splits=original_splits,
                use_filtered_test=use_filtered_test
            )
        
        if args.remove_molecule_leak:
            current_splits, molecule_stats = remove_molecule_leaks(
                current_splits,
                original_splits=original_splits,
                use_filtered_test=use_filtered_test
            )
    
    # 統計情報を保存
    save_statistics(
        args.output_dir,
        initial_splits,
        current_splits,
        mmp_stats,
        frag_stats,
        tid_filter_stats,
        molecule_stats
    )
    
    # PKL ファイルとして保存
    save_cv_splits_as_pkl(current_splits, args.pkl_output)
    
    # dataset_cv{i}.pt ファイルを作成
    create_dataset_pt_files(current_splits, args.data_path, args.output_dir)
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
