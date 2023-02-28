#!/bin/sh
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o output.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
source ~/myEnv/bin/activate
cd ~/data/anair27/singh-lab-TCGA-project
python -u Run.py
