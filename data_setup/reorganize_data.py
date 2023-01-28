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
"""
patient tab path: path to clinical data (this can be retrieved by downloading 'clinical' data from the cart on GDC)
sample sheet path: path to sample sheet (this can be retrieved by downloading
'sample sheet' from the cart on GDC)    
"""

origin_path = ""
sample_sheet_path = ""
destination_path = ""
def reorganize_data(origin_path: str, sample_sheet_path: str, destination_path: str):
    ## Create a directory at the destination
    if not os.path.isdir(destination_path):
        print("Generating new data path.")
        os.mkdir(destination_path)
    else:
        print("New data path already exists.")
    ## In the new directory, add folders for each case
    # Read cases
    sample_data = pd.read_csv(sample_sheet_path, sep = "\t")
    modalities = get_modalities(sample_data)
    cases = get_cases(sample_data)
    print("Adding case folders.")
    for case in cases:
        path = destination_path + "/"+ case
        print("Creating path", path)
        try:
            os.mkdir(path)
            mk_modalities_folders(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            count += 1
            print("Successfully created directory %s " % path)

def get_modalities(sample_data):
    return

def get_cases(sample_data):
    return

def mk_modalities_folders(path, modalities):
    for modality in modalities:
        p = os.join(path, modality)
        os.mkdir(p)


# dbc = DataByCases(patient_tab_path="/users/anair27/data/anair27/data_original/clinical.tsv", sample_sheet_path="/users/anair27/data/anair27/data_original/luad_sample_sheet_09_22.tsv")
# print("CASES:", dbc.cases)

# dbc.reorganize(new_data_path="/users/anair27/data/anair27/data_by_cases")
# dbc.move_data()
