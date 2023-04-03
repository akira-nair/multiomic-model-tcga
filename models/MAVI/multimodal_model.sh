#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -o multimodal%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=akira_nair@brown.edu
# conda activate /users/anair27/anaconda/akira_conda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/anair27/anaconda/akira_conda/lib/
# conda init bash
# conda activate /users/anair27/anaconda/akira_conda

# Parse the arguments passed in by script1.sh
ARGS=("$@")
N=${#ARGS[@]}

# Run the Python script with the arguments

python3 models/MAVI/multimodal_model.py "${ARGS[@]}"