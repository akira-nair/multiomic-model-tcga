#!/usr/bin/env python
'''
File        :   ae_versus_vae.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   This script is an analysis of how VAEs compare AEs for feature reduction
'''

import MAVI
import unimodal_model as um
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import tensorflow as tf

# FIXED HYPERPARAMETERS
EPOCHS_UNSUPERVISED = 100
HIDDEN_DIM = 512
LATENT_DIMS = [8]
LEARNING_RATE_UNSUPERVISED = 0.001
EPOCHS_SUPERVISED = 50
LEARNING_RATE_SUPERVISED = 0.0001
# Modalities
CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
CNV = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv"
EPIGENOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv"
TRANSCRIPTOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv"

# for RNA-seq data
clinical_df = pd.read_csv(CLINICAL)
transcriptomic_df = pd.read_csv(TRANSCRIPTOMIC)
diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
data = transcriptomic_df.merge(diagnosis, on = "case_id").drop_duplicates()
train_df, test_df = train_test_split(data.iloc[:,2:].dropna(), test_size=0.1, random_state=42)

x_train = normalize(train_df.drop('vital_status_Dead', axis = 1))
x_test = normalize(test_df.drop('vital_status_Dead', axis = 1))
y_train = train_df['vital_status_Dead']
y_test = test_df['vital_status_Dead']

for LATENT_DIM in LATENT_DIMS:
    ## vae
    tf.keras.backend.clear_session()
    vae = MAVI.VAE(input_dim = x_train.shape[1], 
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,   
        activation = 'relu',
        lr = LEARNING_RATE_UNSUPERVISED)
    vae.train(x_train, n_epochs= EPOCHS_UNSUPERVISED)
    vae.plot_latent_space(x_train, y_train, output=f'/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/RNA_SEQ_VAE_UMAP_{LATENT_DIM}dim.png')
    x_train_vae_embed = vae.get_embedding(x_train)
    x_test_vae_embed = vae.get_embedding(x_test)
    # unimodal model
    tf.keras.backend.clear_session()
    model_on_vae = um.create_model(x_train = x_train_vae_embed, y_train = y_train, n_epochs = EPOCHS_SUPERVISED, batch_size = None, lr = LEARNING_RATE_SUPERVISED, n_hidden = 1, dim_hidden = LATENT_DIM / 2)
    um.plot_confusion(model_on_vae, x_test_vae_embed, y_test, filepath = f'/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/RNA_SEQ_CM_MODEL_{LATENT_DIM}dim.png')
    ## ae
    tf.keras.backend.clear_session()
    ae = um.create_ae(x_train = x_train, latent_dim = LATENT_DIM, epochs= EPOCHS_UNSUPERVISED, lr = LEARNING_RATE_UNSUPERVISED)
    x_train_ae_embed = um.encode_input(ae, x_train)
    x_test_ae_embed = um.encode_input(ae, x_test)
    tf.keras.backend.clear_session()
    model_on_ae = um.create_model(x_train = x_train_ae_embed, y_train = y_train, n_epochs = EPOCHS_SUPERVISED, batch_size = None, lr = LEARNING_RATE_SUPERVISED, n_hidden = 1, dim_hidden = LATENT_DIM / 2)
    um.plot_confusion(model_on_ae, x_test_ae_embed, y_test, filepath = f'/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/RNA_SEQ_CM_MODEL_{LATENT_DIM}dim.png')
    

    

