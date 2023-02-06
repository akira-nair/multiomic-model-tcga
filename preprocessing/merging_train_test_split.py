#!/usr/bin/env python
'''
File        :   merging_modalities.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Merges processed modality datasets, keeping
                cases that contain all relevant modalities
'''
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

def find_common_cases(modalities: list, index_name = "case_id"):
    common_cases = []
    for i, modality in enumerate(modalities):
        df = pd.read_csv(modality)
        if i == 0:
            common_cases += df["case_id"].tolist()
            print(f"Modality {os.path.basename(modality)} has {len(common_cases)} cases.")
        else:
            common_cases = intersection(common_cases, df["case_id"].tolist())
            print(f"Modality {os.path.basename(modality)} reduced the common cases to {len(common_cases)}")
    return common_cases

def intersection(l1, l2):
    return [item for item in l1 if item in l2]

def save_split(train, test, output_dir: str):
    train_df = pd.DataFrame(train, columns = ['case_id'])
    test_df = pd.DataFrame(test, columns = ['case_id'])
    train_df.to_csv(os.path.join(output_dir, 'training_cases.csv'))
    test_df.to_csv(os.path.join(output_dir, 'testing_cases.csv'))

def main(seed: int):
    CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
    CNV = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv"
    EPIGENOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv"
    TRANSCRIPTOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv"
    IMAGES = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_images_data.csv"
    OUTPUT_DIR = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed"
    common_cases = find_common_cases([CLINICAL, CNV, EPIGENOMIC, TRANSCRIPTOMIC, IMAGES])
    train, test = train_test_split(common_cases, train_size = 0.85, random_state= seed)
    save_split(train, test, OUTPUT_DIR)


if __name__ == "__main__":
    main(sys.argv[1:])