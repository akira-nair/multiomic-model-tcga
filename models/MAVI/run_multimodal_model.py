#!/usr/bin/env python
'''
File        :   run_multimodal_model.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Creates a full multimodal model using
                all the modalities. 
'''

import multimodal_model as mm

import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
# tf.config.experimental.set_visible_devices([], 'GPU')

print(tf.config.list_physical_devices())

OUTPUT = "/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/multimodal_models"
modalities = {
"CNV" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv",
"EPIGENOMIC" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv",
"TRANSCRIPTOMIC" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv",
"IMAGING" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data_updated"
}
training_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/training_cases.csv")["case_id"])
testing_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/testing_cases.csv")["case_id"])


mmm = mm.MultimodalModel(modalities)

CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
clinical_df = pd.read_csv(CLINICAL, index_col = 0)
diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
clinical_data = clinical_df.loc[:, clinical_df.columns != 'vital_status_Dead']

mmm.add_modality(name = "CLINICAL", data = clinical_data)

x_train, y_train, x_test, y_test = mmm.merge_data(y = diagnosis, train_ids = training_cases, test_ids = testing_cases, load_data = True)

mmm.create_model()
tf.keras.utils.plot_model(model = mmm.model, to_file = os.path.join(OUTPUT, "architecture.png"), show_shapes=True, show_layer_names=True, show_layer_activations=True)


model, history = mmm.train_model(n_epochs=20)
model.save(os.path.join(OUTPUT, "mm_model"))
mmm.test_model(os.path.join(OUTPUT, "testing"))