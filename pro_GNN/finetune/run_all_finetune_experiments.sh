#!/bin/bash
# 全ファインチューニング実験を一括実行するスクリプト
#
# 使用例:
#   bash run_all_finetune_experiments.sh
#   bash run_all_finetune_experiments.sh --dry-run   # 実行コマンドを表示のみ

# エラー時に停止するが、算術演算の0は許容
set -eE

# 引数処理
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# ディレクトリ設定
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# 実験設定
TARGETS=("BACE" "JNK1" "P38" "thrombin" "PTP1B" "CDK2")
FOLDS=(0 1 2 3 4)
DATA_MODES=("All" "N100")
MODEL_MODES=("full" "ft")

# カウンター
TOTAL=0
COMPLETED=0
SKIPPED=0
FAILED=0

# 全実験をカウント
for target in "${TARGETS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        for data_mode in "${DATA_MODES[@]}"; do
            for model_mode in "${MODEL_MODES[@]}"; do
                TOTAL=$((TOTAL + 1))
            done
        done
    done
done

echo "========================================"
echo "Finetuning Experiments Batch Runner"
echo "========================================"
echo "Targets:      ${TARGETS[*]}"
echo "Folds:        ${FOLDS[*]}"
echo "Data Modes:   ${DATA_MODES[*]}"
echo "Model Modes:  ${MODEL_MODES[*]}"
echo "Total experiments: ${TOTAL}"
echo "========================================"

# 実験実行
CURRENT=0
for target in "${TARGETS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        for data_mode in "${DATA_MODES[@]}"; do
            for model_mode in "${MODEL_MODES[@]}"; do
                CURRENT=$((CURRENT + 1))
                
                OUTPUT_DIR="results/${data_mode}/${target}/cv${fold}/${model_mode}"
                RESULT_FILE="${OUTPUT_DIR}/test_predictions.npz"
                
                echo ""
                echo "[${CURRENT}/${TOTAL}] ${target} / cv${fold} / ${data_mode} / ${model_mode}"
                
                # 既に完了しているかチェック
                if [ -f "${RESULT_FILE}" ]; then
                    echo "  → Already completed, skipping."
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi
                
                # コマンド
                CMD="bash run_finetune_experiment.sh ${target} ${fold} ${data_mode} ${model_mode}"
                
                if [ "$DRY_RUN" = true ]; then
                    echo "  → Would run: ${CMD}"
                else
                    echo "  → Running..."
                    if ${CMD}; then
                        COMPLETED=$((COMPLETED + 1))
                        echo "  → ✅ Success"
                    else
                        FAILED=$((FAILED + 1))
                        echo "  → ❌ Failed"
                    fi
                fi
            done
        done
    done
done

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Total:     ${TOTAL}"
echo "Completed: ${COMPLETED}"
echo "Skipped:   ${SKIPPED}"
echo "Failed:    ${FAILED}"
echo "========================================"
