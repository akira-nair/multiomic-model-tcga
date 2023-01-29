#!/usr/bin/env python
'''
File        :   merging_modalities.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Merges processed modality datasets, keeping
                cases that contain all relevant modalities
'''
import pandas as pd
import os

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

def main():
    CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
    CNV = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv"
    EPIGENOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv"
    TRANSCRIPTOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv"
    find_common_cases([CLINICAL, CNV, EPIGENOMIC, TRANSCRIPTOMIC])

if __name__ == "__main__":
    main()