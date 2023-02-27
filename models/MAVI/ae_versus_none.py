#!/usr/bin/env python
'''
File        :   ae_versus_none.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   This script explores how an autoencoder enhances 
                the predictive capability of the model
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
sns.set(rc={'figure.figsize':(12,8), 'figure.dpi': 300})   
sns.set_theme(style='white') 
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
norm = preprocessing.MinMaxScaler()
ros = RandomOverSampler(random_state=42)


## Loop through latent dims for imaging modality (no unsupervised learning used for imaging)
def ae_versus_none(modality: str, x_train, x_test, y_train, y_test): 
    output = {}
    x_train = norm.fit_transform(x_train)
    x_test = norm.fit_transform(x_test)
    for LATENT_DIM in LATENT_DIMS:
        ## ae
        tf.keras.backend.clear_session()
        ae = um.create_ae(x_train = x_train, latent_dim = LATENT_DIM, epochs= EPOCHS_UNSUPERVISED, lr = LEARNING_RATE_UNSUPERVISED)
        tf.keras.utils.plot_model(
            ae,
            to_file=os.path.join(OUTPUT_PLOTS, modality, f'AE_{LATENT_DIM}dim.png'),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=False,
            dpi=200,
            show_layer_activations=True
        )
        opt_curves = pd.DataFrame({'training_loss':ae.history.history['loss']})
        opts = sns.lineplot(opt_curves)
        opts.get_figure().savefig(os.path.join(OUTPUT_PLOTS, modality, f'AE_OPTIMIZATION_{LATENT_DIM}dim.png'))
        plt.clf()
        x_train_ae_embed = um.encode_input(ae, x_train)
        x_test_ae_embed = um.encode_input(ae, x_test)
        tf.keras.backend.clear_session()
        model_on_ae, _ = um.create_model(x_train = x_train_ae_embed, y_train = y_train, n_epochs = EPOCHS_SUPERVISED, batch_size = None, lr = LEARNING_RATE_SUPERVISED, n_hidden = 1, dim_hidden = LATENT_DIM / 2)
        tf.keras.utils.plot_model(
            model_on_ae,
            to_file=os.path.join(OUTPUT_PLOTS, modality, f'WITH_AE_MODEL_{LATENT_DIM}dim.png'),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=False,
            dpi=200,
            show_layer_activations=True
        )
        opt_curves = pd.DataFrame({'training_loss':model_on_ae.history.history['loss'], 'val_loss':model_on_ae.history.history['val_loss']})
        opts = sns.lineplot(opt_curves)
        opts.get_figure().savefig(os.path.join(OUTPUT_PLOTS, modality, f'MODEL_ON_AE_OPTIMIZATION_{LATENT_DIM}dim.png'))
        plt.clf()
        with_ae_acc = um.plot_confusion(model_on_ae, x_test_ae_embed, y_test, filepath = os.path.join(OUTPUT_PLOTS, modality, f'CM_AE_REDUCED_MODEL_{LATENT_DIM}dim.png'))
        output[f"With FR to {LATENT_DIM} accuracy"] = with_ae_acc 
        plt.clf()
        ## no ae
        tf.keras.backend.clear_session()
        model_simple, _ = um.create_model(x_train = x_train, y_train = y_train, n_epochs = EPOCHS_SUPERVISED, batch_size = None, lr = LEARNING_RATE_SUPERVISED, n_hidden = 1, dim_hidden = LATENT_DIM / 2)
        tf.keras.utils.plot_model(
            model_simple,
            to_file=os.path.join(OUTPUT_PLOTS, modality, f'WITHOUT_AE_MODEL_{LATENT_DIM}dim.png'),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=False,
            dpi=200,
            show_layer_activations=True
        )
        opt_curves = pd.DataFrame({'training_loss':model_simple.history.history['loss'], 'val_loss':model_simple.history.history['val_loss']})
        opts = sns.lineplot(opt_curves)
        opts.get_figure().savefig(os.path.join(OUTPUT_PLOTS, modality, f'MODEL_SIMPLE_OPTIMIZATION_{LATENT_DIM}dim.png'))
        plt.clf()
        without_ae_acc = um.plot_confusion(model_simple, x_test, y_test, filepath = os.path.join(OUTPUT_PLOTS, modality, f'CM_NO_REDUCTION_MODEL_{LATENT_DIM}dim.png'))
        output[f"Without FR to {LATENT_DIM} accuracy"] = without_ae_acc
        plt.clf()
    return output

# ------ #

# for RNA-seq data
training_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/training_cases.csv")["case_id"])
testing_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/testing_cases.csv")["case_id"])
clinical_df = pd.read_csv(CLINICAL)
transcriptomic_df = pd.read_csv(TRANSCRIPTOMIC)
diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
data = transcriptomic_df.merge(diagnosis, on = "case_id").drop_duplicates()
data_training = data[data["case_id"].isin(training_cases)]
data_testing = data[data["case_id"].isin(testing_cases)]
x_train_raw = data_training.loc[:, data.columns != "vital_status_dead"].iloc[:,2:]
x_test = data_testing.loc[:, data.columns != "vital_status_dead"].iloc[:,2:]
y_train_raw = data_training["vital_status_Dead"].astype(int)
y_test = data_testing["vital_status_Dead"].astype(int)

x_train, y_train = ros.fit_resample(x_train_raw, y_train_raw)
output = ae_versus_none('transcriptomic', x_train, x_test, y_train, y_test)
with open(os.path.join(OUTPUT_PLOTS,'transcriptomic_results.csv'), 'w') as f:
    for key in output:
        f.write("%s,%s\n"%(key,output[key]))

# for CNV data
clinical_df = pd.read_csv(CLINICAL)
cnv_df = pd.read_csv(CNV)
diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
data = cnv_df.merge(diagnosis, on = "case_id").drop_duplicates()
data_training = data[data["case_id"].isin(training_cases)]
data_testing = data[data["case_id"].isin(testing_cases)]
x_train_raw = data_training.loc[:, data.columns != "vital_status_dead"].iloc[:,1:]
x_test = data_testing.loc[:, data.columns != "vital_status_dead"].iloc[:,1:]
y_train_raw = data_training["vital_status_Dead"].astype(int)
y_test = data_testing["vital_status_Dead"].astype(int)
x_train, y_train = ros.fit_resample(x_train_raw, y_train_raw)
output = ae_versus_none('cnv',  x_train, x_test, y_train, y_test)
with open(os.path.join(OUTPUT_PLOTS,'cnv_results.csv'), 'w') as f:
    for key in output:
        f.write("%s,%s\n"%(key,output[key]))

# for Epigenomic data
clinical_df = pd.read_csv(CLINICAL)
epigenomic_df = pd.read_csv(EPIGENOMIC)
diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
data = epigenomic_df.merge(diagnosis, on = "case_id").drop_duplicates()
data_training = data[data["case_id"].isin(training_cases)]
data_testing = data[data["case_id"].isin(testing_cases)]
x_train_raw = data_training.loc[:, data.columns != "vital_status_dead"].iloc[:,2:]
x_test = data_testing.loc[:, data.columns != "vital_status_dead"].iloc[:,2:]
y_train_raw = data_training["vital_status_Dead"].astype(int)
y_test = data_testing["vital_status_Dead"].astype(int)
x_train, y_train = ros.fit_resample(x_train_raw, y_train_raw)
output = ae_versus_none('epigenomic',  x_train, x_test, y_train, y_test)
with open(os.path.join(OUTPUT_PLOTS,'epigenomic_results.csv'), 'w') as f:
    for key in output:
        f.write("%s,%s\n"%(key,output[key]))
###
###
print("Completed AE VERSUS NONE .py")