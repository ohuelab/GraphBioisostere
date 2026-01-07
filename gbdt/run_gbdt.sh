#!/bin/sh
#$ -cwd
#$ -l cpu_80=1
#$ -l h_rt=4:00:00
#$ -N run_gbdt
#$ -m abe
#$ -M masunaga.s.aa@m.titech.ac.jp
#$ -j y
#$ -o tmp/

. /etc/profile.d/modules.sh
source $HOME/.bashrc


conda activate bioiso

# 実行したい設定のリスト (2048, 1024, frag)
for CONFIG_SUFFIX in 2048 frag; do
    # 各設定に対して、0から4までのfoldでループ
    for FOLD in 0 1 2 3 4; do
        date
        # CONFIG_SUFFIXを使って、入力ファイルと出力ディレクトリのパスを動的に作成
        INPUTFILE="/home/8/uf02678/gsbsmasunaga/MMP_dataset/dataset_consistentsmiles-${CONFIG_SUFFIX}.joblib"
        OUTPUT_DIR="/home/8/uf02678/gsbsmasunaga/bioiso/gbdt/results_consistentsmiles-${CONFIG_SUFFIX}"
        PKL_FILE="/home/8/uf02678/gsbsmasunaga/bioiso/splitting/tid_5cv_consistentsmiles_molecule.pkl"

        echo "Running GBDT with config ${CONFIG_SUFFIX}, fold $FOLD"
        echo "Input file: ${INPUTFILE}"
        echo "Output dir: ${OUTPUT_DIR}"

        python run_gbdt.py --input_file $INPUTFILE --output_dir $OUTPUT_DIR --pkl_file $PKL_FILE --fold $FOLD --force
        date
    done
done
echo done
