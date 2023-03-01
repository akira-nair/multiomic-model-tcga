#!/usr/bin/env python
'''
File        :   reorganize_data.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Given a download and sample sheet, organize data by cases
(where each case gets its own folder)
'''
import pandas as pd
import os
import shutil
import sys
"""
patient tab path: path to clinical data (this can be retrieved by downloading 'clinical' data from the cart on GDC)
sample sheet path: path to sample sheet (this can be retrieved by downloading
'sample sheet' from the cart on GDC)    
"""
# example ** PLEASE UPDATE IN MAIN IF NOT USING SYSTEM ARGS **
origin_path = "/users/anair27/data/TCGA_Data/project_DLBC/data_original"
sample_sheet_path = "/users/anair27/data/TCGA_Data/project_DLBC/sample_sheet_DLBC.tsv"
destination_path = "/users/anair27/data/TCGA_Data/project_DLBC/data_by_cases"

def reorganize_data(origin_path: str, sample_sheet_path: str, destination_path: str):
    # specify the subdirectory name for each modality of interest
    type_extension = {
        "Gene Expression Quantification": "gene_expression",
        "Gene Level Copy Number": "cnv",
        "Slide Image": "images",
        "Methylation Beta Value": "dna_methylation"
    }
    ## Create a directory at the destination
    if not os.path.isdir(destination_path):
        print("Generating new data path.")
        os.mkdir(destination_path)
    else:
        print("New data path already exists.")
    ## In the new directory, add folders for each case
    # Read cases
    sample_data = pd.read_csv(sample_sheet_path, sep = "\t")
    modalities = list(type_extension.values())
    cases = get_cases(sample_data)
    print("Adding case folders.")
    for case in cases:
        path = destination_path + "/"+ case
        print("Creating path", path)
        try:
            os.mkdir(path)
            mk_modalities_folders(path, modalities)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created directory %s " % path)
    count = 0
    for index,row in sample_data.iterrows():
        assoc_case = row['Case ID'].split(",")[0].strip()
        assoc_type = row['Data Type']
        assoc_file = row['File ID']
        assoc_file_name = row['File Name']
        source = os.path.join(origin_path, assoc_file, assoc_file_name)
        if assoc_type in type_extension:
            dest = os.path.join(destination_path, assoc_case, type_extension[assoc_type])
        else:
            dest = os.path.join(destination_path, assoc_case, "other")
        print("Moving", source, "to", dest)
        try:
            shutil.move(source, dest)
            count +=1
        except:
            print("Error in moving data from original data to new data")
            print("Filename: " + assoc_case + " - " + assoc_file_name)
    print(f"Moved {count} files")

def get_cases(sample_data):
    cases_raw = sample_data['Case ID'].values.tolist()
    cases_processed = set()
    for case in cases_raw:
        cases_processed.add(case.split(",")[0].strip())
    return list(cases_processed)

def mk_modalities_folders(path, modalities):
    for modality in modalities:
        p = os.path.join(path, modality)
        os.mkdir(p)
    os.mkdir(os.path.join(path, "other"))
   

def main(argv):
    if len(argv) > 0:
        origin_path = argv[0]
        sample_sheet_path = argv[1]
        destination_path = argv[2]
    else:
        # defaults
        # DEFAULT EXAMPLE (change to your filepath!)
        origin_path = "/users/anair27/data/TCGA_Data/project_DLBC/data_original"
        sample_sheet_path = "/users/anair27/data/TCGA_Data/project_DLBC/sample_sheet_DLBC.tsv"
        destination_path = "/users/anair27/data/TCGA_Data/project_DLBC/data_by_cases"
    reorganize_data(origin_path = origin_path, sample_sheet_path = sample_sheet_path, destination_path = destination_path)
if __name__ == "__main__":
    main(sys.argv[1:])
# dbc = DataByCases(patient_tab_path="/users/anair27/data/anair27/data_original/clinical.tsv", sample_sheet_path="/users/anair27/data/anair27/data_original/luad_sample_sheet_09_22.tsv")
# print("CASES:", dbc.cases)

# dbc.reorganize(new_data_path="/users/anair27/data/anair27/data_by_cases")
# dbc.move_data()
