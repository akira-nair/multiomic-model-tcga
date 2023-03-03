#!/usr/bin/env python
'''
File        :   imaging_unimodal.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Creates multiple unimodal imaging models
'''


import unimodal_model as um
import importlib as imp
from PIL import Image
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import unimodal_model as um
import random
import seaborn as sns
import matplotlib.pyplot as plt
from output_folder import create_output_folder


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    sns.set()
    sns.set(rc={'figure.figsize':(12,8), 'figure.dpi': 300})   
    sns.set_theme(style='white') 
    SEEDS = [42, 15, 0, 1, 67, 128, 87, 261, 510, 340, 22]
    # SEEDS = [42, 15, 0]
    IMAGE = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data_updated"
    CLINICAL = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv"
    OUTPUT = "/users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/__plots/"
    subdirs = [str(s) for s in SEEDS]
    OUTPUT_FOLDER = create_output_folder("imaging_models", subdirs = subdirs)
    training_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/training_cases.csv")["case_id"])
    testing_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/testing_cases.csv")["case_id"])
    example_case = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data_updated/TCGA-69-7978.jpeg"
    example_image = Image.open(example_case)
    image_size = np.array(example_image).shape
    X_train = np.empty((len(training_cases),) + image_size, dtype='float32')
    y_train = np.empty((len(training_cases),), dtype='int')
    X_test = np.empty((len(testing_cases),) + image_size, dtype='float32')
    y_test = np.empty((len(testing_cases),), dtype='int')
    clinical_df = pd.read_csv(CLINICAL)
    diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
    # Load and convert the images for training
    for i, case_id in enumerate(training_cases):
        # Load the image file
        image_file = os.path.join(IMAGE, f'{case_id}.jpeg')
        image = Image.open(image_file)

        # Resize the image and convert it to the output format
        # image = image.resize(image_size)
        image = np.array(image, dtype='float32')
        image /= 255.0

        # Store the image and its label in the output data structure
        X_train[i] = image
        y_train[i] = (int)(diagnosis[diagnosis["case_id"] == case_id]["vital_status_Dead"])
    
    # Load and convert the images for testing
    for i, case_id in enumerate(testing_cases):
        # Load the image file
        image_file = os.path.join(IMAGE, f'{case_id}.jpeg')
        image = Image.open(image_file)

        # Resize the image and convert it to the output format
        # image = image.resize(image_size)
        image = np.array(image, dtype='float32')
        image /= 255.0

        # Store the image and its label in the output data structure
        X_test[i] = image
        y_test[i] = (int)(diagnosis[diagnosis["case_id"] == case_id]["vital_status_Dead"])
    t_accs = []
    v_accs = []
    for s in SEEDS:
        tf.keras.backend.clear_session()
        reset_random_seeds(s)
        model, hist = um.create_model_image(X_train, y_train, image_shape=image_size, \
                                    n_hidden = 1, lr = 0.01, optimizer= 'adam', n_epochs = 20, batch_size = None,\
                                    balance_class=True)
        v_acc = model.history.history['val_accuracy'].pop()
        opt_curves = pd.DataFrame({'training_loss':model.history.history['loss'], 'val_loss':model.history.history['val_loss']})
        opts = sns.lineplot(opt_curves)
        opts.get_figure().savefig(os.path.join(OUTPUT_FOLDER, str(s), "convergence.png"))
        plt.clf()
        t_acc = um.plot_confusion(model, X_test, y_test, os.path.join(OUTPUT_FOLDER, str(s), "test_cm.png"))
        t_accs.append(t_acc)
        v_accs.append(v_acc)
        plt.clf()
    results = pd.DataFrame({'validation':v_accs,'testing': t_accs, 'seed': subdirs})
    results.to_csv(os.path.join(OUTPUT_FOLDER, "results.csv"))
    strip = sns.stripplot(x="variable", y="value", hue="seed", data=pd.melt(results, id_vars=["seed"]), jitter=False, size = 10)
    plt.ylim(0.5, 1)
    strip.get_figure().savefig(os.path.join(OUTPUT_FOLDER, "accuracies_across_seeds_stripplot.png"))
    plt.clf()
    box = sns.boxplot(x="variable", y="value", data=pd.melt(results, id_vars=["seed"]))
    plt.ylim(0.5, 1)
    box.get_figure().savefig(os.path.join(OUTPUT_FOLDER, "accuracies_across_seeds_boxplot.png"))
plt.ylim(0.5, 1)

if __name__ == '__main__':
    main()
    print("\n\n**COMPLETE**\n\n")