#!/bin/sh
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o hp_tuning_%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
# conda activate /users/anair27/anaconda/akira_conda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/anair27/anaconda/akira_conda/lib/
# argument 1, modality; argument 2, number of trials
# sbatch models/MAVI/hp_tuning.sh "epigenomic" 30
# sbatch models/MAVI/hp_tuning.sh "transcriptomic" 30
# sbatch models/MAVI/hp_tuning.sh "cnv" 30
python3 models/MAVI/um_hyperparameter_tuning.py ${1} ${2}