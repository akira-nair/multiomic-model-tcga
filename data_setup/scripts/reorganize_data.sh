#!/bin/sh
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o reorganize_data.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
source ~/myEnv/bin/activate
cd /users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/data_setup
python3 -u reorganize_data.py ${1} ${2} ${3}
