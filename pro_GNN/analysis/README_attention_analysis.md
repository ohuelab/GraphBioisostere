# Attention Visualization and Structure Analysis

このディレクトリには、分子ペアのアテンション可視化と構造解析のための2つのスクリプトが含まれています。

## ファイル構成

### 1. `visualize_attention.py`
- **目的**: 少数のペアに対するアテンション可視化
- **出力**: SVGファイル、詳細情報テキストファイル
- **用途**: 個別ペアの詳細な視覚的分析

### 2. `analyze_attention_structure.py`
- **目的**: 大量データに対するアテンション-構造関係の統計解析
- **出力**: 統計レポート、CSVファイル
- **用途**: 全体的な傾向の把握、研究論文用の統計データ

## 基本的な使用方法

### アテンション可視化 (少数ペア)

```bash
# 基本的な使用 - 全標的から10ペアをランダムサンプリング
python visualize_attention.py \
    --model_path results_0907/cv0/pair-diff/model_best.pt \
    --csv_path ../../unimols/dataset.csv \
    --n_pairs 10 \
    --output_dir attention_visualization

# 特定の標的を指定してサンプリング
python visualize_attention.py \
    --model_path results_0907/cv0/pair-diff/model_best.pt \
    --csv_path ../../unimols/dataset.csv \
    --target_id CHEMBL312 \
    --n_pairs 5 \
    --output_dir attention_visualization_chembl312
```

### 統計解析 (大量データ)

```bash
# 全データの統計解析
python analyze_attention_structure.py \
    --model_path results_0907/cv0/pair-diff/model_best.pt \
    --csv_path ../../unimols/dataset.csv \
    --output_dir attention_structure_analysis

# 特定標的の統計解析
python analyze_attention_structure.py \
    --model_path results_0907/cv0/pair-diff/model_best.pt \
    --csv_path ../../unimols/dataset.csv \
    --target_id CHEMBL312 \
    --max_pairs 1000 \
    --output_dir attention_analysis_chembl312
```

## 標的指定機能

### 利用可能な標的の確認

```bash
# 利用可能な標的一覧を表示
python visualize_attention.py \
    --csv_path ../../unimols/dataset.csv \
    --list_targets

# 統計解析スクリプトでも同様
python analyze_attention_structure.py \
    --csv_path ../../unimols/dataset.csv \
    --list_targets
```

### 標的指定のオプション

- `--target_id`: 特定の標的ID (例: CHEMBL312, CHEMBL3247)
- `--target_col`: 標的IDのカラム名 (デフォルト: TID)
- `--list_targets`: 利用可能な標的一覧を表示

## 主要なオプション

### 共通オプション
- `--model_path`: 学習済みモデルのパス
- `--csv_path`: データセットCSVファイルのパス
- `--output_dir`: 結果出力ディレクトリ
- `--target_id`: 対象標的ID
- `--random_seed`: 再現性のためのランダムシード

### 可視化スクリプト固有
- `--n_pairs`: サンプリングするペア数 (デフォルト: 5)
- `--ref_col`: REF-SMILESカラム名 (デフォルト: REF-SMILES)
- `--prb_col`: PRB-SMILESカラム名 (デフォルト: PRB-SMILES)

### 統計解析スクリプト固有
- `--max_pairs`: 処理する最大ペア数 (デフォルト: 制限なし)
- `--batch_size`: バッチ処理サイズ (デフォルト: 100)

## 出力ファイル

### 可視化スクリプトの出力
- `pair_N/ref_molecule_attention.svg`: REF分子のアテンション可視化
- `pair_N/prb_molecule_attention.svg`: PRB分子のアテンション可視化
- `pair_N/pair_summary.svg`: ペア全体の統合可視化
- `pair_N/detailed_info.txt`: 詳細メタデータ
- `analysis_summary.txt`: 全体レポート

### 統計解析スクリプトの出力
- `attention_structure_statistics.txt`: 統計レポート
- `attention_statistics_detailed.csv`: 全ペアの詳細データ
- `attention_statistics_summary.csv`: 標的別サマリー

## HPC環境での実行

### qsubスクリプトの使用

```bash
# ジョブスクリプトを実行
qsub run_attention_analysis.sh

# 標的を指定したジョブ
# run_attention_analysis.shを編集して--target_id CHEMBL312を追加
```

### バッチジョブの例

```bash
# 複数の標的を順次処理
for target in CHEMBL312 CHEMBL3247 CHEMBL2056; do
    python analyze_attention_structure.py \
        --model_path results_0907/cv0/pair-diff/model_best.pt \
        --csv_path ../../unimols/dataset.csv \
        --target_id $target \
        --output_dir attention_analysis_$target
done
```

## 研究での活用方法

### 1. 探索的解析
```bash
# まず利用可能な標的を確認
python visualize_attention.py --csv_path ../../unimols/dataset.csv --list_targets

# 興味のある標的のペア数とラベル分布を確認
python visualize_attention.py \
    --csv_path ../../unimols/dataset.csv \
    --target_id CHEMBL312 \
    --n_pairs 10
```

### 2. 詳細可視化
```bash
# 特定標的の代表的なペアを可視化
python visualize_attention.py \
    --model_path results_0907/cv0/pair-diff/model_best.pt \
    --csv_path ../../unimols/dataset.csv \
    --target_id CHEMBL312 \
    --n_pairs 20 \
    --random_seed 42 \
    --output_dir figures_chembl312
```

### 3. 統計的解析
```bash
# 標的別の包括的統計解析
python analyze_attention_structure.py \
    --model_path results_0907/cv0/pair-diff/model_best.pt \
    --csv_path ../../unimols/dataset.csv \
    --target_id CHEMBL312 \
    --output_dir stats_chembl312

# 全標的の比較解析
python analyze_attention_structure.py \
    --model_path results_0907/cv0/pair-diff/model_best.pt \
    --csv_path ../../unimols/dataset.csv \
    --max_pairs 5000 \
    --output_dir stats_all_targets
```

## トラブルシューティング

### よくあるエラー

1. **標的が見つからない**
   ```
   ValueError: 標的 'CHEMBL999' のデータが見つかりません
   ```
   → `--list_targets` で利用可能な標的を確認してください

2. **メモリ不足**
   → `--batch_size` を小さくするか、`--max_pairs` で制限してください

3. **CUDA エラー**
   → CPUモードで実行するか、より小さなバッチサイズを使用してください

### パフォーマンス最適化

- **大量データ処理**: `batch_size=50` 程度に設定
- **メモリ制約**: `max_pairs=1000` で制限
- **並列処理**: 複数の標的を別々のジョブで処理

## 応用例

### 標的別アテンションパターンの比較
複数の標的で統計解析を実行し、アテンションパターンの違いを比較する研究に活用できます。

### 薬物設計への応用
特定の標的で共通部分と置換部分のどちらに注目すべきかを統計的に把握し、創薬研究の指針とします。

### モデル解釈性の研究
生物学的等価体予測において、モデルが何に注目しているかを定量的に解析します。