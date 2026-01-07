#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=06:00:00
#$ -N job
#$ -m abe
#$ -M masunaga.s.aa@m.titech.ac.jp

. /etc/profile.d/modules.sh
source $HOME/.bashrc
conda activate bioiso
module load intel-mpi

INPUTFILE=$1
OUTPUTDIR=$2
loss_type=$3
model_type=$4

# gpu_logger log gpu_$JOB_NAME-$JOB_ID &

echo $split $loss_type $model_type
# $OUTPUTDIR/test_predictions.npz
if [ -f $OUTPUTDIR/test_predictions.npz ]; then
    echo "File $OUTPUTDIR/test_predictions.npz exists. Exiting."
    exit 0
fi
# Change to the pro_GNN directory to ensure correct relative paths
cd $(dirname $0)/..

torchrun --standalone \
        --nproc-per-node=1 \
training_cls_ddp_3.py --input_file $INPUTFILE --output_dir $OUTPUTDIR \
        --loss_type $loss_type  --model_type $model_type \
        --batch_size 8192 --epochs 300 --patience 30


# python -m torch.distributed.launch \
#   --nproc_per_node=2 \
#   --master_port=6123 \
#   training_cls_ddp_2.py \
#   --input_dir $INPUTDIR --output_dir $OUTPUTDIR --loss_type $loss_type --patience 50 --model_type $model_type --batch_size 64

date
