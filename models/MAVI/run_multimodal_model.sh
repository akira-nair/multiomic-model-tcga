#!/bin/sh
#SBATCH --mem=4G
#SBATCH -t 0:10:00
#SBATCH -o multimodal_complete.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
# conda activate /users/anair27/anaconda/akira_conda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/anair27/anaconda/akira_conda/lib/
# conda init bash
# conda activate /users/anair27/anaconda/akira_conda

# module load openssl/3.0.0 cuda/11.7.1 cudnn/8.6.0 gcc/10.2
# conda activate akira_conda
# # CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
# python3 models/MAVI/run_multimodal_model.py
# Generate a string with the current date and time in ISO 8601 format
DATE=$(date +%Y-%m-%d_%H-%M-%S)

# Create a directory with the current date and time as its name
mkdir __plots/multimodal_model_"$DATE"

# modalities = {
# "CNV" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv",
# "EPIGENOMIC" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv",
# "TRANSCRIPTOMIC" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv",
# "CLINICAL" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data_no_target.csv",
# "IMAGING" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data_updated"
# }
# Define the list of mods and filepaths
MODS=('CNV' 'EPIGENOMIC' 'TRANSCRIPTOMIC' 'CLINICAL' 'IMAGING')
FILEPATHS=('/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data_no_target.csv' '/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data_updated')

# Get the length of the lists
N=${#MODS[@]}


for ((i=2;i<=${#mods[@]};i++)); do
  for subset in $(compgen -A variable | grep -oP "(?<=mod)\d+" | xargs -I{} echo {\#mod\[\]} | tr ' ' '\n' | xargs -n $i echo); do
    mod_args=()
    path_args=()
    for j in $subset; do
      mod_args+=("${mods[j-1]}")
      path_args+=("${paths[j-1]}")
    done
    sbatch models/MAVI/multimodal_model.sh -m "${mod_args[@]}" -f "${path_args[@]}" -o __plots/multimodal_model_"$DATE"
  done
done

# Loop over all possible subsets of the list
for ((i=0; i < 2**N; i++)); do
    # Convert the loop counter to a binary string and pad with zeros
    BINARY=$(echo "obase=2; $i" | bc)
    PADDED_BINARY=$(printf "%0*d" $N $BINARY)

    # Create the subsets using the binary string
    MOD_SUBSET=()
    FILEPATH_SUBSET=()
    for ((j=0; j < N; j++)); do
        if [ ${PADDED_BINARY:$j:1} == "1" ]; then
            MOD_SUBSET+=(${MODS[$j]})
            FILEPATH_SUBSET+=(${FILEPATHS[$j]})
        fi
    done

    # Call sbatch on another shell script with the members of the subsets as parameters
    sbatch models/MAVI/multimodal_model.sh -m "${MOD_SUBSET[@]}" -f "${FILEPATH_SUBSET[@]}" -o __plots/multimodal_model_"$DATE"
done