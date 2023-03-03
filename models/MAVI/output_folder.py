#!/usr/bin/env python
'''
File        :   output_folder.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Creates an output folder for a script
'''
import datetime
import os
def create_output_folder(job_name, subdirs = ["cnv", "epigenomic", "transcriptomic", "imaging"]):
    timestamp = datetime.datetime.now().strftime("%h-%d-%H:%M:%S")
    OUTPUT = os.path.join('/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/', f"{job_name}-{timestamp}")
    os.mkdir(OUTPUT)
    for subdir in subdirs:
        os.mkdir(os.path.join(OUTPUT, subdir))
    return OUTPUT