#!/bin/sh
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
# cd into the directory to download data
cd ~/data/anair27/data_original/
# go to where the gdc client is downloaded, and specify the manifest file in the final argument
~/data/anair27/data_original/gdc-client download -m  ~/data/anair27/data_original/luad_manifest_09_22.txt
