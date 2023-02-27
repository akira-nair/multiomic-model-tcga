#!/bin/sh
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -o ae_versus_none.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
# conda activate /users/anair27/anaconda/akira_conda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/anair27/anaconda/akira_conda/lib/
python3 models/MAVI/ae_versus_none.py