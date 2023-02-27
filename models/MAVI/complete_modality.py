#!/usr/bin/env python
'''
File        :   complete_modality.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   This script compares five modalities with sets of hyperparameters
                and shows how autoencoders as a method for feature reduction
                can help strengthen performance
'''
import MAVI
import unimodal_model as um
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
sns.set()

# FIXED HYPERPARAMETERS
EPOCHS_UNSUPERVISED = 150
HIDDEN_DIM = 512
LATENT_DIMS = [32, 64, 256]
LEARNING_RATE_UNSUPERVISED = 0.001
EPOCHS_SUPERVISED = 100
LEARNING_RATE_SUPERVISED = 0.001

CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
CNV = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv"
EPIGENOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv"
TRANSCRIPTOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv"
IMAGING = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data"

# Make output plot directories 
timestamp = datetime.datetime.now().strftime("%h-%d-%H:%M:%S")
OUTPUT_PLOTS = os.path.join('/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/', timestamp)
os.mkdir(OUTPUT_PLOTS)
os.mkdir(os.path.join(OUTPUT_PLOTS, "cnv"))
os.mkdir(os.path.join(OUTPUT_PLOTS, "epigenomic"))
os.mkdir(os.path.join(OUTPUT_PLOTS, "transcriptomic"))
os.mkdir(os.path.join(OUTPUT_PLOTS, "imaging"))
norm = preprocessing.MinMaxScaler()
ros = RandomOverSampler(random_state=42)

