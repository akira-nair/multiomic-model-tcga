#!/bin/sh
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o create_images_preprocessing.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
# conda activate /users/anair27/anaconda/akira_conda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/anair27/anaconda/akira_conda/lib/
python3 preprocessing/preprocess_images/create_images_dataset.py