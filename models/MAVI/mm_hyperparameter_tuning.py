#!/usr/bin/env python
'''
File        :   run_multimodal_model.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Creates a full multimodal model using
                all the modalities. 
'''

import argparse
from seed import reset_random_seeds
import multimodal_model as mm
import plots
import pandas as pd
import os
import optuna
import sys
import copy
import logging
import joblib
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from output_folder import create_output_folder
# tf.config.experimental.set_visible_devices([], 'GPU')

# print(tf.config.list_physical_devices())

# OUTPUT = "/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/multimodal_models"


def optimization(trials: int, attention: bool, load_study: str = None, feature_reduction: bool = False):
    
    modalities = {
    "CNV" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_cnv_data.csv",
    "EPIGENOMIC" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_epigenomic_data.csv",
    "TRANSCRIPTOMIC" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_transcriptomic_data.csv",
    "CLINICAL" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data_no_target.csv",
    "IMAGING" : "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data_updated"
    }
    training_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/training_cases.csv")["case_id"])
    testing_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/testing_cases.csv")["case_id"])

    hyperparameters = {
        "learning_rate" : 0.001,
        "n_epochs" : 20,
        "batch_size" : 30,
        "CNV" : {
            "dim_hidden":512,
            "num_hidden":3,
            "dim_end":1024
        },
        "EPIGENOMIC" : {
            "dim_hidden":512,
            "num_hidden":3,
            "dim_end":1024
        },
        "TRANSCRIPTOMIC" : {
            "dim_hidden":512,
            "num_hidden":3,
            "dim_end":1024
        },
        "CLINICAL" : {
            "dim_hidden":16,
            "num_hidden":2,
            "dim_end":32
        },
        "IMAGING" : {
            "dim_hidden":16,
            "num_hidden":2,
            "dim_end":32
        },
    }
    OUTPUT = create_output_folder("hyperparameter-tuning", range(trials))
    logger_file = os.path.join(OUTPUT, "logger.log")
    logging.basicConfig(filename=logger_file, filemode='w', encoding='utf-8', \
        level=logging.INFO)
    # # Create a logger object
    # logger = logging.getLogger('my_logger')

    # # Create a file handler that logs messages to a file

    # log_file_path = os.path.join(OUTPUT, "logger.out")
    # file_handler = logging.FileHandler(log_file_path)

    # # Set the logging level of the file handler to INFO
    # file_handler.setLevel(logging.INFO)

    # # Create a formatter that specifies the log message format
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)

    # # Add the file handler to the logger
    # logger.addHandler(file_handler)
    # logger.setLevel(logging.INFO)
    logging.info("Hyperparameter tuning")
    logging.info(f"GPU: {tf.config.list_physical_devices('GPU')}")

    CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
    clinical_df = pd.read_csv(CLINICAL, index_col = 0)
    diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
    reset_random_seeds(42)
    if attention:
        logging.info("Using attention model.")
    if feature_reduction:
        logging.info("Using feature reduction.")
    mmm = mm.MultimodalModel(modalities, attention=attention)
    logging.info(f"GPU: {tf.test.gpu_device_name()}")
    logging.info(tf.__version__)
    mmm.merge_data(y = diagnosis, train_ids = training_cases, test_ids = testing_cases, load_data = True)
    def objective(trial):
        reset_random_seeds(42)
        
        hyperparameters = {
        "learning_rate" : trial.suggest_categorical("learning_rate", [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
        "n_epochs" : trial.suggest_int("n_epochs", 5, 40, 5), 
        "batch_size" : trial.suggest_categorical("batch_size_supervised", [16, 32, 64]),
        "CNV" : {
            "dim_hidden": trial.suggest_categorical("dim_hidden_CNV", [32, 64, 128, 256, 512, 1024, 4096]),
            "num_hidden": trial.suggest_int("num_hidden_CNV", 1, 5, 1),
            "dim_end": trial.suggest_categorical("dim_end_CNV", [32, 256, 512, 1024, 4096, 16384]),
            "feature_reduction": feature_reduction
        },
        "EPIGENOMIC" : {
            "dim_hidden": trial.suggest_categorical("dim_hidden_EPIGENOMIC", [32, 64, 128, 256, 512, 1024, 4096]),
            "num_hidden": trial.suggest_int("num_hidden_EPIGENOMIC", 1, 5, 1),
            "dim_end": trial.suggest_categorical("dim_end_EPIGENOMIC", [32, 256, 512, 1024, 4096, 16384]),
            "feature_reduction": feature_reduction
        },
        "TRANSCRIPTOMIC" : {
            "dim_hidden": trial.suggest_categorical("dim_hidden_TRANSCRIPTOMIC", [32, 64, 128, 256, 512, 1024, 4096]),
            "num_hidden": trial.suggest_int("num_hidden_TRANSCRIPTOMIC", 1, 5, 1),
            "dim_end": trial.suggest_categorical("dim_end_TRANSCRIPTOMIC", [32, 256, 512, 1024, 4096, 16384]),
            "feature_reduction": feature_reduction
        },
        "CLINICAL" : {
            "dim_hidden": trial.suggest_categorical("dim_hidden_CLINICAL", [32, 64, 128]),
            "num_hidden": trial.suggest_int("num_hidden_CLINICAL", 1, 5, 1),
            "dim_end": trial.suggest_categorical("dim_end_CLINICAL", [8, 16, 32]),
            "feature_reduction": False
        },
        "IMAGING" : {
            "dim_hidden":16,
            "num_hidden": trial.suggest_int("num_hidden_IMAGING", 1, 5, 1),
            "dim_end":32,
            "num_convs":trial.suggest_int("num_convs_IMAGING", 1, 5, 1),
            "dropout":trial.suggest_categorical("dropout_IMAGING", [True, False]),
            "feature_reduction": False
        }
        }
        mmm.set_hyperparameters(hyperparameters)
        logging.info(f"\n\nHyperparameters for trial {(str)(trial.number)}: {hyperparameters} \n")
        trial_dir = os.path.join(OUTPUT, (str)(trial.number))
        if not os.path.exists(trial_dir):
            os.mkdir(trial_dir)
        mmm.create_model()
        logging.info(f"Plotting model...")
        tf.keras.utils.plot_model(model = mmm.model, to_file = os.path.join(trial_dir, "architecture.png"), show_shapes=True, show_layer_names=True, show_layer_activations=True)
        logging.info(f"Training model...")
        model, hist = mmm.train_model()
        logging.info(f"Model trained. \n\n\n\n")
        history = copy.deepcopy(hist)
        model.save(os.path.join(trial_dir, "mm_model"))
        mmm.test_model(os.path.join(trial_dir, "testing"))
        plots.plot_convergence(history, os.path.join(trial_dir, "convergence"))
        return history.history["val_accuracy"].pop()
    # now create optimization study   
    if load_study is None: 
        logging.info(f"Creating a new study")
        study = optuna.create_study(study_name = "mm_hpt_study", direction = "maximize")
    else:
        logging.info(f"Using study at {load_study}")
        study = joblib.load(load_study)
    study.optimize(objective, n_trials=trials)
    df_results = study.trials_dataframe()
    df_results.to_csv(os.path.join(OUTPUT, "study_results.csv"))
    joblib.dump(study, os.path.join(OUTPUT, "mm_hpt_study.pkl"))
    logging.info('--------------------------------')
    logging.info("Hyperparameter optimization results:")
    logging.info(f"BEST PARAMETERS {study.best_params} HAD BEST VALUE {study.best_value} \n\n Complete.")
    return study

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--trials', type=int, help='Number of trials for hyperparameter tuning', default=3)
    parser.add_argument('-f', '--filepath', type=str, help='Optional filepath to load existing study')
    parser.add_argument('-a', '--attention', action='store_true', help='Flag to indicate whether to use attention or not')
    parser.add_argument('-r', '--reducedims', action='store_true', help='Flag to indicate whether to reduce dimensions using a RF classifier')

    args = parser.parse_args(argv)
    if args.filepath:
        optimization(trials = args.trials, attention= args.attention,load_study=args.filepath, feature_reduction = args.reducedims)
    else:
        optimization(trials = args.trials, attention = args.attention, feature_reduction = args.reducedims)
    


if __name__ == '__main__':
    main(sys.argv[1:])
# clinical_data = clinical_df.loc[:, clinical_df.columns != 'vital_status_Dead']

#mmm.add_modality(name = "CLINICAL", data = clinical_data)




