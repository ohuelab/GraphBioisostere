#!/bin/bash
# 単一のファインチューニング実験を実行するスクリプト
#
# 使用例:
#   bash run_finetune_experiment.sh BACE 0 All ft
#   bash run_finetune_experiment.sh JNK1 2 N100 full

set -e

# 引数の確認
if [ $# -lt 4 ]; then
    echo "Usage: $0 <target> <fold> <data_mode> <model_mode> [pretrain_dir]"
    echo ""
    echo "Arguments:"
    echo "  target:       Target name (BACE, JNK1, P38, thrombin, PTP1B, CDK2)"
    echo "  fold:         Fold number (0-4)"
    echo "  data_mode:    Data mode (All or N100 for limited training)"
    echo "  model_mode:   Model mode (full or ft for finetuning)"
    echo "  pretrain_dir: Pretrained model directory (optional, default: results_consistentsmiles_tid3_molecule)"
    exit 1
fi

TARGET=$1
FOLD=$2
DATA_MODE=$3
MODEL_MODE=$4
PRETRAIN_DIR=${5:-"results_consistentsmiles_tid3_molecule"}

# ディレクトリ設定
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="${SCRIPT_DIR}"  # finetune ディレクトリ
PRO_GNN_DIR="$(dirname "${SCRIPT_DIR}")"  # pro_GNN ディレクトリ
DATA_DIR="${BASE_DIR}/target_datasets"  # ターゲットデータ (prepare_target_datasets.pyで作成)
RESULTS_DIR="${BASE_DIR}/results"  # 結果保存先
PRETRAIN_MODEL_DIR="${PRO_GNN_DIR}/results/${PRETRAIN_DIR}"  # 事前学習モデル

# 入力ファイル
INPUT_FILE="${DATA_DIR}/${TARGET}/dataset_cv${FOLD}.pt"

# 出力ディレクトリ
OUTPUT_DIR="${RESULTS_DIR}/${DATA_MODE}/${TARGET}/cv${FOLD}/${MODEL_MODE}"

# 事前学習モデルのパス（ftモードの場合）
PRETRAIN_MODEL="${PRETRAIN_MODEL_DIR}/cv${FOLD}/pair-cat/model_best.pt"

# 確認
echo "========================================"
echo "Finetuning Experiment"
echo "========================================"
echo "Target:       ${TARGET}"
echo "Fold:         ${FOLD}"
echo "Data Mode:    ${DATA_MODE}"
echo "Model Mode:   ${MODEL_MODE}"
echo "Input File:   ${INPUT_FILE}"
echo "Output Dir:   ${OUTPUT_DIR}"
if [ "${MODEL_MODE}" == "ft" ]; then
    echo "Pretrain:     ${PRETRAIN_MODEL}"
fi
echo "========================================"

# 入力ファイルの存在確認
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: Input file not found: ${INPUT_FILE}"
    echo "Please run prepare_target_datasets.py first."
    exit 1
fi

# 事前学習モデルの確認（ftモードの場合）
if [ "${MODEL_MODE}" == "ft" ]; then
    if [ ! -f "${PRETRAIN_MODEL}" ]; then
        echo "Error: Pretrained model not found: ${PRETRAIN_MODEL}"
        exit 1
    fi
fi

# 出力ディレクトリ作成
mkdir -p "${OUTPUT_DIR}"

# トレーニングサイズの設定
TRAINING_SIZE_ARG=""
if [ "${DATA_MODE}" == "N100" ]; then
    TRAINING_SIZE_ARG="--training_size 100"
elif [ "${DATA_MODE}" == "N50" ]; then
    TRAINING_SIZE_ARG="--training_size 50"
fi

# 事前学習モデルの引数
PRETRAIN_ARG=""
FREEZE_ARG=""
LR="0.0001"

if [ "${MODEL_MODE}" == "ft" ]; then
    PRETRAIN_ARG="--pretrain_model ${PRETRAIN_MODEL}"
    FREEZE_ARG="--freeze_encoder"  # fine-tuning時はエンコーダを固定
    LR="0.00001"  # fine-tuning時は学習率を1/10に
fi

# 実行
cd "${BASE_DIR}"

python "${BASE_DIR}/finetune_reg.py" \
    --input_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 64 \
    --epochs 100 \
    --lr ${LR} \
    --patience 10 \
    --embedding_dim 64 \
    --hidden_dim 64 \
    --num_layers 3 \
    --dropout 0.1 \
    --model_type cat \
    --seed 42 \
    ${TRAINING_SIZE_ARG} \
    ${PRETRAIN_ARG} \
    ${FREEZE_ARG}

echo ""
echo "✅ Experiment completed: ${OUTPUT_DIR}"
