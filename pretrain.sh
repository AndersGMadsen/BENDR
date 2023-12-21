#!/bin/bash
#SBATCH --job-name=BENDR
#SBATCH --output=~/BENDR_ORIGINAL/logs/output_%J.out
#SBATCH --error=~/BENDR_ORIGINAL/logs/error_%J.err
#SBATCH --gres=gpu:Turing:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=96gb

#SBATCH --nodelist=comp-gpu02

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

## SCRATCH=/scratch/$USER
## if [[ ! -d $SCRATCH ]]; then
##   mkdir $SCRATCH
## fi

WORKDIR=~/BENDR_ORIGINAL
cd "$WORKDIR"

source ~/.bashrc
module load CUDA/12.1 CUDNN/8.8
conda activate BENDR
python pretrain.py --config configs/pretraining.yml --results-folder "results"

echo "Done: $(date +%F-%R:%S)"