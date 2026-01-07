#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N job_bioiso
#$ -m abe
#$ -M masunaga.s.aa@m.titech.ac.jp
#$ -o tmp/
#$ -e tmp/

. /etc/profile.d/modules.sh
source $HOME/.bashrc
conda activate bioiso
module load intel-mpi

# mode : diff, cat, product
# Bioisostere prediction
for cvnum in 0 1 2 3 4; do
    for losstype in "pair"; do
        for mode in product;do
            INPUTFILE=dataset/dataset_consistentsmiles_molecule/dataset_cv${cvnum}.pt
            OUTPUTDIR=results/results_consistentsmiles_molecule/cv${cvnum}/$losstype-$mode/
            echo miqsub -N bio-$cvnum-$losstype-$mode
            local/run_ddp.sh $INPUTFILE $OUTPUTDIR $losstype $mode
            sleep 1
        done
    done
done

# # Transfer learning
# for target in  BACE JNK1 P38 CDK2 thrombin PTP1B;do
#     miqsub  local/run_fine.sh $target
#     sleep 1
# done
