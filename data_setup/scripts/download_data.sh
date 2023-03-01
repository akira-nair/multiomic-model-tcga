#!/bin/sh
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o download_data.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
# cd into the directory to download data
cd ~/data/TCGA_Data/project_DLBC/data_original
# go to where the gdc client is downloaded, and specify the manifest file in the final argument
/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/gdc-client download -m  ~/data/TCGA_Data/project_DLBC/manifest_DLBC.txt
