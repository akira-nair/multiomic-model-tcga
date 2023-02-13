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
LATENT_DIMS = [32, 256]
LEARNING_RATE_UNSUPERVISED = 0.001
EPOCHS_SUPERVISED = 100
LEARNING_RATE_SUPERVISED = 0.001
# Modalities
CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
CNV = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv"
EPIGENOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv"
TRANSCRIPTOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv"
# MODALITY_TESTING = 'TRANSCRIPTOMIC'
# Make output plot directories 
timestamp = datetime.datetime.now().strftime("%h-%d-%H:%M:%S")
OUTPUT_PLOTS = os.path.join('/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/', timestamp)
os.mkdir(OUTPUT_PLOTS)
os.mkdir(os.path.join(OUTPUT_PLOTS, "cnv"))
os.mkdir(os.path.join(OUTPUT_PLOTS, "epigenomic"))
os.mkdir(os.path.join(OUTPUT_PLOTS, "transcriptomic"))
norm = preprocessing.MinMaxScaler()
ros = RandomOverSampler(random_state=42)

def ae_versus_vae(modality: str, x_train, x_test, y_train, y_test): 
    x_train = norm.fit_transform(x_train)
    x_test = norm.fit_transform(x_test)
    # x_train = norm.fit_transform(train_df.drop('vital_status_Dead', axis = 1))
    # x_test = norm.fit_transform(test_df.drop('vital_status_Dead', axis = 1))
    # y_train = train_df['vital_status_Dead']
    # y_test = test_df['vital_status_Dead']
    for LATENT_DIM in LATENT_DIMS:
        ## vae
        tf.keras.backend.clear_session()
        vae = MAVI.VAE(input_dim = x_train.shape[1], 
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,   
            activation = 'sigmoid',
            lr = LEARNING_RATE_UNSUPERVISED)
        vae.train(x_train, n_epochs= EPOCHS_UNSUPERVISED)
        opt_curves = pd.DataFrame({'training_loss':vae.history.history['loss'], 'val_loss':vae.history.history['val_loss']})
        opts = sns.lineplot(opt_curves)
        opts.get_figure().savefig(os.path.join(OUTPUT_PLOTS, modality, f'VAE_OPTIMIZATION_{LATENT_DIM}dim.png'))
        plt.clf()
        vae.plot_latent_space(x_train, y_train, output=os.path.join(OUTPUT_PLOTS, modality, f'VAE_UMAP_{LATENT_DIM}dim.png'))
        plt.clf()
        x_train_vae_embed = vae.get_embedding(x_train)
        x_test_vae_embed = vae.get_embedding(x_test)
        # unimodal model
        tf.keras.backend.clear_session()
        model_on_vae, _ = um.create_model(x_train = x_train_vae_embed, y_train = y_train, n_epochs = EPOCHS_SUPERVISED, batch_size = None, lr = LEARNING_RATE_SUPERVISED, n_hidden = 1, dim_hidden = LATENT_DIM / 2)
        opt_curves = pd.DataFrame({'training_loss':model_on_vae.history.history['loss'], 'val_loss':model_on_vae.history.history['val_loss']})
        opts = sns.lineplot(opt_curves)
        opts.get_figure().savefig(os.path.join(OUTPUT_PLOTS, modality, f'MODEL_ON_VAE_OPTIMIZATION_{LATENT_DIM}dim.png'))
        plt.clf()
        um.plot_confusion(model_on_vae, x_test_vae_embed, y_test, filepath = os.path.join(OUTPUT_PLOTS, modality, f'CM_FULL_MODEL_VAE_{LATENT_DIM}dim.png'))
        plt.clf()
        ## ae
        tf.keras.backend.clear_session()
        ae = um.create_ae(x_train = x_train, latent_dim = LATENT_DIM, epochs= EPOCHS_UNSUPERVISED, lr = LEARNING_RATE_UNSUPERVISED)
        opt_curves = pd.DataFrame({'training_loss':ae.history.history['loss']})
        opts = sns.lineplot(opt_curves)
        opts.get_figure().savefig(os.path.join(OUTPUT_PLOTS, modality, f'AE_OPTIMIZATION_{LATENT_DIM}dim.png'))
        plt.clf()
        x_train_ae_embed = um.encode_input(ae, x_train)
        x_test_ae_embed = um.encode_input(ae, x_test)
        tf.keras.backend.clear_session()
        model_on_ae, _ = um.create_model(x_train = x_train_ae_embed, y_train = y_train, n_epochs = EPOCHS_SUPERVISED, batch_size = None, lr = LEARNING_RATE_SUPERVISED, n_hidden = 1, dim_hidden = LATENT_DIM / 2)
        opt_curves = pd.DataFrame({'training_loss':model_on_ae.history.history['loss'], 'val_loss':model_on_ae.history.history['val_loss']})
        opts = sns.lineplot(opt_curves)
        opts.get_figure().savefig(os.path.join(OUTPUT_PLOTS, modality, f'MODEL_ON_AE_OPTIMIZATION_{LATENT_DIM}dim.png'))
        plt.clf()
        um.plot_confusion(model_on_ae, x_test_ae_embed, y_test, filepath = os.path.join(OUTPUT_PLOTS, modality, f'CM_FULL_MODEL_AE_{LATENT_DIM}dim.png'))
        plt.clf()

# for RNA-seq data
clinical_df = pd.read_csv(CLINICAL)
transcriptomic_df = pd.read_csv(TRANSCRIPTOMIC)
diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
data = transcriptomic_df.merge(diagnosis, on = "case_id").drop_duplicates()
x = data.loc[:, data.columns != "vital_status_dead"]
y = data["vital_status_Dead"].astype(int)
x_ros, y_ros = ros.fit_resample(x, y)
X_train, X_test, y_train, y_test = train_test_split(x_ros.iloc[:, 2:].dropna(), y_ros, test_size=0.1, random_state=42)
ae_versus_vae('transcriptomic', X_train, X_test, y_train, y_test)

# for CNV data
clinical_df = pd.read_csv(CLINICAL)
cnv_df = pd.read_csv(CNV)
diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
data = cnv_df.merge(diagnosis, on = "case_id").drop_duplicates()
x = data.loc[:, data.columns != "vital_status_dead"]
y = data["vital_status_Dead"].astype(int)
x_ros, y_ros = ros.fit_resample(x, y)
X_train, X_test, y_train, y_test = train_test_split(x_ros.iloc[:, 1:].dropna(), y_ros, test_size=0.1, random_state=42)
ae_versus_vae('cnv',  X_train, X_test, y_train, y_test)

# for Epigenomic data
clinical_df = pd.read_csv(CLINICAL)
epigenomic_df = pd.read_csv(EPIGENOMIC)
diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
data = epigenomic_df.merge(diagnosis, on = "case_id").drop_duplicates()
x = data.loc[:, data.columns != "vital_status_dead"]
y = data["vital_status_Dead"].astype(int)
x_ros, y_ros = ros.fit_resample(x, y)
X_train, X_test, y_train, y_test = train_test_split(x_ros.iloc[:, 2:].dropna(), y_ros, test_size=0.1, random_state=42)
ae_versus_vae('epigenomic',  X_train, X_test, y_train, y_test)



# TODO: add the plot directory as an output


    

    

