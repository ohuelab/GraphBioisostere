#!/usr/bin/env python3
"""
元データから各ターゲット用のデータセットを作成するスクリプト

元データ: /home/8/uf02678/gsbsmasunaga/MMP_dataset/dataset.csv
出力先: bioiso/notebooks/target/{TARGET}/target_df.csv
"""

import os
import pandas as pd
from pathlib import Path
import argparse

# ターゲット情報（ChEMBL ID とターゲット名のマッピング）
TARGET_INFO = {
    "CHEMBL4822": "BACE",
    "CHEMBL301": "CDK2",
    "CHEMBL2276": "JNK1",
    "CHEMBL4361": "MCL1",
    "CHEMBL260": "P38",
    "CHEMBL335": "PTP1B",
    "CHEMBL204": "thrombin",
    "CHEMBL3553": "TYK2"
}


def load_original_dataset(data_path: Path) -> pd.DataFrame:
    """
    元のデータセットを読み込む
    
    Args:
        data_path: dataset.csvへのパス
        
    Returns:
        DataFrame
    """
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total pairs: {len(df)}")
    print(f"Unique targets: {df['TID'].nunique()}")
    return df


def create_target_dataset(
    df: pd.DataFrame,
    target_id: str,
    target_name: str,
    output_dir: Path
):
    """
    特定ターゲットのデータセットを作成
    
    Args:
        df: 元のDataFrame
        target_id: ターゲットのChEMBL ID
        target_name: ターゲット名
        output_dir: 出力先ディレクトリ
    """
    # ターゲットでフィルタ
    target_df = df[df['TID'] == target_id].copy()
    
    if len(target_df) == 0:
        print(f"⚠️  No data found for {target_name} (ID: {target_id})")
        return
    
    print(f"\n=== Processing {target_name} (ID: {target_id}) ===")
    print(f"Total pairs: {len(target_df)}")
    
    # 必要なカラムのみを抽出
    required_columns = [
        'REF-CID', 'PRB-CID', 'TID',
        'REF-SMILES', 'PRB-SMILES',
        'REF-standard_value', 'PRB-standard_value',
        'delta_value', 'STANDARD_TYPE'
    ]
    
    # 存在するカラムのみを選択
    available_columns = [col for col in required_columns if col in target_df.columns]
    target_df = target_df[available_columns]
    
    # 出力ディレクトリ作成
    target_output_dir = output_dir / target_name
    target_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存
    output_file = target_output_dir / "target_df.csv"
    target_df.to_csv(output_file, index=False)
    
    print(f"Saved: {output_file}")
    print(f"  - Unique REF molecules: {target_df['REF-SMILES'].nunique()}")
    print(f"  - Unique PRB molecules: {target_df['PRB-SMILES'].nunique()}")
    
    # 統計情報
    print(f"  - Delta value range: [{target_df['delta_value'].min():.3f}, {target_df['delta_value'].max():.3f}]")
    print(f"  - Delta value mean: {target_df['delta_value'].mean():.3f} ± {target_df['delta_value'].std():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Create target-specific datasets")
    parser.add_argument("--input_file", type=str,
                        default="/home/8/uf02678/gsbsmasunaga/MMP_dataset/dataset.csv",
                        help="Path to original dataset.csv")
    parser.add_argument("--output_dir", type=str,
                        default=None,
                        help="Output directory (default: bioiso/notebooks/target)")
    parser.add_argument("--targets", type=str, nargs="+",
                        default=None,
                        help="Target IDs to process (default: all)")
    
    args = parser.parse_args()
    
    # パス設定
    input_file = Path(args.input_file)
    
    if args.output_dir is None:
        # スクリプトの場所から相対パスで設定
        script_dir = Path(__file__).parent
        bioiso_dir = script_dir.parent.parent
        output_dir = bioiso_dir / "notebooks" / "target"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("Create Target-Specific Datasets")
    print("="*50)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # データ読み込み
    df = load_original_dataset(input_file)
    
    # ターゲットIDの確認（指定されたターゲットのみ確認）
    available_targets = set(df['TID'].unique())
    
    # 処理対象のターゲットを決定
    if args.targets is None:
        # デフォルト: TARGET_INFOに定義されているターゲットのみ
        targets_to_process = [(tid, name) for tid, name in TARGET_INFO.items() if tid in available_targets]
    else:
        # 指定されたターゲットのみ
        targets_to_process = []
        for target_spec in args.targets:
            # TID（CHEMBL ID）で指定された場合
            if target_spec.startswith("CHEMBL"):
                name = TARGET_INFO.get(target_spec, target_spec)
                targets_to_process.append((target_spec, name))
            # ターゲット名で指定された場合
            else:
                matching_tids = [k for k, v in TARGET_INFO.items() if v == target_spec]
                if matching_tids:
                    tid = matching_tids[0]
                    targets_to_process.append((tid, target_spec))
                else:
                    print(f"Warning: Unknown target specification: {target_spec}")
    
    print(f"\nProcessing {len(targets_to_process)} targets...")
    
    # 各ターゲットのデータセットを作成
    for target_id, target_name in targets_to_process:
        create_target_dataset(df, target_id, target_name, output_dir)
    
    print("\n" + "="*50)
    print("All target datasets created!")
    print("="*50)


if __name__ == "__main__":
    main()
