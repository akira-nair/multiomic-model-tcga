#!/bin/sh
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o hp_tuning.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
# conda activate /users/anair27/anaconda/akira_conda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/anair27/anaconda/akira_conda/lib/
python3 models/MAVI/um_hyperparameter_tuning.py ${1} ${2}