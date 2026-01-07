#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N baseline_bioiso
#$ -m abe
#$ -M masunaga.s.aa@m.titech.ac.jp
#$ -o tmp/
#$ -e tmp/

. /etc/profile.d/modules.sh
source $HOME/.bashrc
conda activate bioiso
module load intel-mpi

# Baseline GNN model experiments
# encoder_type : gin, gcn, gin_edge
# mode : diff, cat, product
# losstype : pair, pair_bi, pair_bi_sym

for cvnum in 0 1 2 3 4; do
    for encoder in gcn; do
        for losstype in "pair"; do
            for mode in cat; do
                INPUTFILE=dataset/dataset_consistentsmiles_tid3_molecule/dataset_cv${cvnum}.pt
                OUTPUTDIR=results/baseline_${encoder}_consistentsmiles_tid3_molecule/cv${cvnum}/$losstype-$mode/
                echo miqsub -N baseline-$encoder-$cvnum-$losstype-$mode
                local/run_baseline_gnn.sh $INPUTFILE $OUTPUTDIR $encoder $losstype $mode
                sleep 1
            done
        done
    done
done
