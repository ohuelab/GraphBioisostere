#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=06:00:00
#$ -N baseline_gnn
#$ -m abe
#$ -M masunaga.s.aa@m.titech.ac.jp

. /etc/profile.d/modules.sh
source $HOME/.bashrc
conda activate bioiso
module load intel-mpi

INPUTFILE=$1
OUTPUTDIR=$2
ENCODER_TYPE=$3  # gin, gcn, gin_edge
loss_type=$4
model_type=$5

# デフォルト値の設定
if [ -z "$ENCODER_TYPE" ]; then
    ENCODER_TYPE="gin"
fi
if [ -z "$loss_type" ]; then
    loss_type="pair"
fi
if [ -z "$model_type" ]; then
    model_type="diff"
fi

# gpu_logger log gpu_$JOB_NAME-$JOB_ID &

echo "Baseline GNN Training"
echo "Encoder Type: $ENCODER_TYPE, Loss Type: $loss_type, Model Type: $model_type"
echo "Input: $INPUTFILE"
echo "Output: $OUTPUTDIR"

# $OUTPUTDIR/test_predictions.npz
if [ -f $OUTPUTDIR/test_predictions.npz ]; then
    echo "File $OUTPUTDIR/test_predictions.npz exists. Exiting."
    exit 0
fi

# Change to the pro_GNN directory to ensure correct relative paths
cd $(dirname $0)/..

torchrun --standalone \
        --nproc-per-node=1 \
training_baseline_gnn.py --input_file $INPUTFILE --output_dir $OUTPUTDIR \
        --encoder_type $ENCODER_TYPE --loss_type $loss_type --model_type $model_type \
        --batch_size 8192 --epochs 300 --patience 30

date
