import optuna
import unimodal_model as um
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from optuna.visualization import plot_optimization_history, plot_contour, plot_parallel_coordinate
import plotly
import sys
import os
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
sns.set()
sns.set(rc={'figure.figsize':(12,8), 'figure.dpi': 300})   
sns.set_theme(style='white') 
timestamp = datetime.datetime.now().strftime("%h-%d-%H:%M:%S")
OUTPUT_PLOTS = os.path.join('/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/', f"{timestamp}-hpt")
os.mkdir(OUTPUT_PLOTS)
os.mkdir(os.path.join(OUTPUT_PLOTS, "cnv"))
os.mkdir(os.path.join(OUTPUT_PLOTS, "epigenomic"))
os.mkdir(os.path.join(OUTPUT_PLOTS, "transcriptomic"))
CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
CNV = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv"
EPIGENOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv"
TRANSCRIPTOMIC = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv"
IMAGING = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data"

ros = RandomOverSampler(random_state=42)

def optimization(x_train, y_train, x_test, y_test, trials, modality, validation_split = 0.1):
    norm = preprocessing.MinMaxScaler()
    x_train = norm.fit_transform(x_train)
    x_test = norm.fit_transform(x_test)
    def objective(trial):
        EPOCHS_UNSUPERVISED = trial.suggest_int("epochs_unsupervised", 20, 200, 30)
        BATCH_SIZE_SUPERVISED = trial.suggest_int("batch_size_supervised", 2, 64, 16)
        HIDDEN_DIM = trial.suggest_int("dim_hidden", 32, 128, 16)
        N_HIDDEN = trial.suggest_int("n_hidden", 1, 4)
        LATENT_DIM = trial.suggest_int("dim_latent", 32, 128, 32)
        LEARNING_RATE_UNSUPERVISED = trial.suggest_float("lr_unsupervised", 0.0001, 0.01)
        EPOCHS_SUPERVISED = trial.suggest_int("epochs_supervised", 20, 200, 30)
        LEARNING_RATE_SUPERVISED = trial.suggest_float("lr_supervised", 0.0001, 0.01)
        
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
        model_on_ae, hist = um.create_model(x_train = x_train_ae_embed, y_train = y_train, n_epochs = EPOCHS_SUPERVISED, batch_size = BATCH_SIZE_SUPERVISED, lr = LEARNING_RATE_SUPERVISED, n_hidden = N_HIDDEN, dim_hidden = HIDDEN_DIM, validation_set=validation_split)
        opt_curves = pd.DataFrame({'training_loss':model_on_ae.history.history['loss'], 'val_loss':model_on_ae.history.history['val_loss']})
        opts = sns.lineplot(opt_curves)
        opts.get_figure().savefig(os.path.join(OUTPUT_PLOTS, modality, f'MODEL_ON_AE_OPTIMIZATION_{LATENT_DIM}dim.png'))
        plt.clf()
        um.plot_confusion(model_on_ae, x_test_ae_embed, y_test, filepath = os.path.join(OUTPUT_PLOTS, modality, f'CM_AE_REDUCED_MODEL_{LATENT_DIM}dim.png'))
        plt.clf()
        print("HISTORY KEYS")
        print(hist.history)
        return hist.history["val_sparse_categorical_accuracy"].pop()

    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials=trials)
    df_results = study.trials_dataframe()
    df_results.to_csv(os.path.join(OUTPUT_PLOTS, "study_results.csv"))
    print('--------------------------------')
    print("Hyperparameter optimization results:")
    print(f"BEST PARAMETERS {study.best_params} HAD BEST VALUE {study.best_value}")
    return study

def plot_results(study):
    plot_history = plot_optimization_history(study)
    plot_cont = plot_contour(study)
    plot_par = plot_parallel_coordinate(study)
    os.chdir(OUTPUT_PLOTS)
    plot_history.write_image(f"./HPT_history_{timestamp}.png")
    plot_cont.write_image(f"./HPT_contour_{timestamp}.png")
    plot_par.write_image(f"./HPT_parallel_{timestamp}.png")

def main(argv):
    training_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/training_cases.csv")["case_id"])
    testing_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/testing_cases.csv")["case_id"])
    modality = argv[0]
    n_trials = int(argv[1])
    if modality == "transcriptomic":
        # for transcriptomic data
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
        optimization(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, trials = n_trials, modality = "transcriptomic")
    elif modality == "cnv":
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
        optimization(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, trials = n_trials, modality = "cnv")
    elif modality == "epigenomic":
        # for epigenomic data
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
        optimization(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, trials = n_trials, modality = "epigenomic")
    else:
        print(f"Modality {modality} is not registered in this script.")
    # X_train = pd.DataFrame(np.load(argv[0]))
    # y_train = pd.DataFrame(np.load(argv[1]))
    # study = optimization(X_train, y_train, int(argv[2]))
    # plot_results(study)

if __name__ == '__main__':
    main(sys.argv[1:])