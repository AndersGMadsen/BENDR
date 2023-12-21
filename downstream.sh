#!/bin/bash
#SBATCH --job-name=BENDR
#SBATCH --output=~/BENDR_ORIGINAL/logs/downstream/output_%J.out
#SBATCH --error=~/BENDR_ORIGINAL/logs/downstream/error_%J.err
#SBATCH --gres=gpu:Turing:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --mem=32gb
##SBATCH --nodelist=comp-gpu02

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

WORKDIR=~/BENDR_ORIGINAL
cd "$WORKDIR"

source ~/.bashrc
module load CUDA/12.1 CUDNN/8.8
conda activate BENDR
python downstream.py BENDR --ds-config configs/downstream.yml --results-filename results/downstream_BENDR.xlsx --metrics-config configs/metrics.yml 
echo "Done: $(date +%F-%R:%S)"
