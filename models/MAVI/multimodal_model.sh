#!/bin/sh
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -o multimodal_model.out
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


# sbatch models/MAVI/multimodal_modal.sh -m 'CNV' 'EPIGENOMIC' 'TRANSCRIPTOMIC' 'CLINICAL' -f '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data_no_target.csv' -o '/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/quadrimodal_model'

# MODS=('CNV' 'EPIGENOMIC' 'TRANSCRIPTOMIC' 'CLINICAL' 'IMAGING')
# FILEPATHS=('/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data_no_target.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data_updated')