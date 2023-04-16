#!/bin/sh
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o hp_tuning_mm.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
# conda activate /users/anair27/anaconda/akira_conda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/anair27/anaconda/akira_conda/lib/
# argument 1, modality; argument 2, number of trials
# sbatch models/MAVI/mm_hp_tuning.sh 10
# sbatch models/MAVI/hp_tuning.sh "transcriptomic" 30
# sbatch models/MAVI/hp_tuning.sh "cnv" 30
#SBATCH -p gpu --gres=gpu:1
module load python/3.9.0 openssl/3.0.0 cuda/11.7.1 cudnn/8.2.0
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda deactivate
conda activate /users/anair27/anaconda/akira_conda
# conda uninstall tensorflow
# conda install tensorflow==2.11.0
# python3 models/MAVI/simple_model.py
# python3 models/MAVI/mm_hyperparameter_tuning.py -t 7 -f ${2} -a
# python3 models/MAVI/mm_hyperparameter_tuning.py -t 7 -a
ARGS=("$@")
python3 models/MAVI/mm_hyperparameter_tuning.py "${ARGS[@]}"
# python3 models/MAVI/mm_hyperparameter_tuning.py -t 3 -f /users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/hyperparameter-tuning-Apr-10-01:06:36/mm_hpt_study.pkl -a
# sbatch models/MAVI/mm_hyperparameter_tuning.sh -t 3 -a -r
# # export PATH=/gpfs/runtime/opt/cuda/11.7.1/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/gpfs/runtime/opt/cuda/11.7.1/cuda/lib64:$LD_LIBRARY_PATH