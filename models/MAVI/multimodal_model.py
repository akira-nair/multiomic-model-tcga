#!/usr/bin/env python
'''
File        :   multimodal_model.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Provides functions to create
                customizable multimodal models
                trained on multiple modalities
'''
import pandas as pd
from PIL import Image
import numpy as np
import os
import plots
import argparse
import sys
from sklearn.utils import compute_class_weight
from sklearn import preprocessing
import os
import tensorflow as tf
from imblearn.over_sampling import ADASYN
import logging
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

class MultimodalModel():
    """
    Multimodal model
    """
    def __init__(self, modalities = None, settings = None, attention = False) -> None:
        self.modalities: list[Modality] = []
        if modalities is not None:
            if isinstance(modalities, list):
                for modality in modalities:
                    self.modalities.append(Modality(name = modality))
            elif isinstance(modalities, dict):
                for modality in modalities:
                    self.modalities.append(Modality(name = modality, data = modalities[modality]))
            else:
                raise TypeError("Modalities should be passed as either a list of strings or a dictionary with modality names corresponding to data paths.")
        if settings is not None:
            self.set_hyperparameters(settings)
        self.settings = settings
        self.model = None
        self.attention = attention
    
    def set_hyperparameters(self, settings: dict):
        self.n_epochs = settings["n_epochs"]
        self.learning_rate = settings["learning_rate"]
        self.batch_size = settings["batch_size"]
        for modality in self.modalities:
            modality_hyperparameters = settings[modality.name]
            modality.set_hyperparameters(modality_hyperparameters)

    def reset(self):
        tf.keras.backend.clear_session()

    def add_modality(self, name, data):
        new_modality = Modality(name, data)
        self.modalities.append(new_modality)

    def merge_data(self, y, merge_on = "case_id", target_name = "vital_status_Dead", train_ids = None, test_ids = None, rebalance_data = False, load_data = False):
        modalities = []
        for modality in self.modalities:
            if modality.is_imaging == False:
                modalities.append(modality)
        if len(modalities) == 1:
            self.merged = modalities[0].data
        else:
            logging.info(f"Merging {modalities[0].name} and {modalities[1].name}...")
            self.merged = modalities[0].data.merge(modalities[1].data, on = merge_on).drop_duplicates()
            if len(modalities) > 2:
                for modality in modalities[2:]:
                    logging.info(f"Merging in {modality.name}...")
                    self.merged = self.merged.merge(modality.data, on = merge_on).drop_duplicates()
        logging.info(f"Merging target variable data (y)...")
        self.merged = self.merged.merge(y, on = merge_on).drop_duplicates()
        if train_ids is not None and test_ids is not None:
            data_training = self.merged[self.merged[merge_on].isin(train_ids)]
            data_testing = self.merged[self.merged[merge_on].isin(test_ids)]
            x_train = data_training.loc[:, self.merged.columns != target_name]
            x_test = data_testing.loc[:, self.merged.columns != target_name]
            self.y_train = data_training[target_name].astype(int)
            self.y_test = data_testing[target_name].astype(int)

            if rebalance_data:
                for modality in self.modalities:
                    if modality.is_imaging:
                        raise InterruptedError("Cannot rebalance data since images are used.")
                
                resampler = ADASYN(random_state=42)
                logging.info("Rebalancing data...")
                x_train = x_train.drop(columns = [merge_on])
                logging.info(f'Original dataset shape had: {Counter(self.y_train)}'), 
                x_train, self.y_train = resampler.fit_resample(x_train, self.y_train)
                logging.info(f'Resampled dataset shape has: {Counter(self.y_train)}')

            if load_data:
                for modality in self.modalities:
                    modality.load_modality_data(x_train, x_test)
            return x_train, self.y_train, x_test, self.y_test
        else:
            return self.merged

    def create_model(self):
        del self.model
        model_inputs_layers = []
        modality_end_layers = []
        if len(self.modalities) == 1:
            modality = self.modalities[0]
            input = tf.keras.layers.Input(shape=(modality.training_data.shape[1], ), name=f"{modality.name}_input_layer")
            modality_end = tf.keras.layers.Dense(256, activation='relu', name=f"{modality.name}_end")(input)
            predictions = tf.keras.layers.Dense(2, activation='sigmoid', name='output')(modality_end)
            self.model: tf.keras.models.Model = tf.keras.models.Model(inputs=input, outputs=predictions)
        else:
            for modality in self.modalities:
                if modality.is_imaging:
                    input = tf.keras.layers.Input(shape = modality.image_size, name=f"{modality.name}_input_layer")
                    prev = input
                    for c in range(modality.num_convs):
                        conv1 = tf.keras.layers.Conv2D(64, (3, 3),  activation='relu')(prev)
                        pool1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)
                        if modality.dropout:
                            pool1 = tf.keras.layers.Dropout(0.1)(pool1)
                        prev = pool1
                    flatten = tf.keras.layers.Flatten()(prev)
                    modality_end = tf.keras.layers.Dense(256, activation='relu', name=f"{modality.name}_end")(flatten)
                    
                else:
                    if modality.feature_reduction:
                        sel = SelectFromModel(RandomForestClassifier(n_estimators = 50, random_state=42))
                        sel.fit(modality.training_data, self.y_train)
                        selected_feat = pd.DataFrame(modality.training_data).columns[(sel.get_support())]
                        logging.info(f"{len(selected_feat)} features were selected for {modality.name} data using a Random Forest Classifier")
                        modality.training_data = modality.training_data[:, sel.get_support()]
                        modality.testing_data = modality.testing_data[:, sel.get_support()]
                    # define input layer
                    input = tf.keras.layers.Input(shape=(modality.training_data.shape[1], ), name=f"{modality.name}_input_layer")
                    norm = tf.keras.layers.BatchNormalization()(input)
                    if modality.feature_reduction == False:
                        prev = norm
                        for h in range(modality.num_hidden):
                            modality_hidden = tf.keras.layers.Dense(modality.dim_hidden, activation='relu', name=f"{modality.name}_hidden_{h}")(prev)
                            norm_modality_hidden = tf.keras.layers.BatchNormalization()(modality_hidden)
                            prev = norm_modality_hidden
                        modality_end = tf.keras.layers.Dense(modality.dim_end, activation='relu', name=f"{modality.name}_end")(modality_hidden)
                    else:
                        modality_end = norm
                model_inputs_layers.append(input)
                modality_end_layers.append(modality_end)
            if self.attention:
                attention_layers = []
                for modality in modality_end_layers:
                    attention_layers.append(self_attention(modality))
                cm_attention_layers = []
                for i in range(len(attention_layers)):
                    for j in range(i + 1, len(attention_layers)):
                        x, y = attention_layers[i], attention_layers[j]
                        cm_attention_layers.append(cross_modal_attention(x, y))
                concat = tf.keras.layers.concatenate(cm_attention_layers)
            else:
                concat = tf.keras.layers.concatenate(modality_end_layers)
            predictions = tf.keras.layers.Dense(2, activation='sigmoid', name='output')(concat)
            self.model: tf.keras.models.Model = tf.keras.models.Model(inputs=model_inputs_layers, outputs=predictions)
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.model.compile(optimizer = optimizer, loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])

    def train_model(self, n_epochs = 10):
        if self.model is None:
            raise AttributeError("Model has not been instantiated. Call create_model() first.")
        training_inputs = [m.training_data for m in self.modalities]
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=list(self.y_train))
        class_weights = {0: class_weights[0], 1: class_weights[1]}    
        self.history: tf.keras.callbacks.History = self.model.fit(training_inputs, tf.keras.utils.to_categorical(self.y_train),
        batch_size=self.batch_size, epochs=self.n_epochs, validation_split=0.1, class_weight=class_weights, verbose=2)
        return self.model, self.history
    
    def test_model(self, output):
        testing_inputs = [m.testing_data for m in self.modalities]
        plots.plot_confusion(self.model, testing_inputs, self.y_test, output)

    def generate_summary(self, output):
        text = f"This is a multimodal model trained on the modalities: {self.modalities}.\n\n"
        text += f"The model trained for {self.n_epochs} epochs, a learning rate of {self.learning_rate}, and a batch size of {self.batch_size}.\n"
        if self.attention:
            text += f"The model was trained with attention.\n"
        else:
            text += f"The model was trained without attention.\n"
        
        with open(output, 'w') as f:
            f.write(text)
            f.write(self.settings)

    def __str__(self) -> str:
        return f"Multimodal model with modalities {self.modalities}"


class Modality():
    def __init__(self, name, data = None, omit_vars = ["case_id", "index"]) -> None:
        self.name: str = name
        self.vars = None
        if data is not None:
            if self.name.lower() in ["image", "imaging"]:
                self.is_imaging = True
                self.data = data
            else:
                self.is_imaging = False
                self.data = data
                if isinstance(self.data, str):
                    self.data: pd.DataFrame = pd.read_csv(data, index_col = 0)
                renamed_cols = {}
                for col in self.data.columns:
                    if col not in omit_vars:
                        renamed_cols[col] = f"{self.name}-{col}"
                self.data = self.data.rename(columns=renamed_cols)
                self.vars = list(set(self.data.columns) - set(omit_vars))

    def set_hyperparameters(self, settings: dict):
        if self.is_imaging:
            self.num_convs = settings["num_convs"]
            self.dropout = settings["dropout"]
        self.dim_hidden = settings["dim_hidden"]
        self.num_hidden = settings["num_hidden"]
        self.dim_end = settings["dim_end"]
        self.feature_reduction = settings["feature_reduction"]

    def load_modality_data(self, merged_data_train, merged_data_test):
        if self.is_imaging:
            training_cases = list(merged_data_train['case_id'])
            testing_cases = list(merged_data_test['case_id'])
            example_case = os.path.join(self.data, os.listdir(self.data)[0])
            example_image = Image.open(example_case)
            image_size = np.array(example_image).shape
            self.image_size = image_size
            x_train = np.empty((len(training_cases),) + image_size, dtype='float32')
            # y_train = np.empty((len(training_cases),), dtype='int')
            x_test = np.empty((len(testing_cases),) + image_size, dtype='float32')
            # y_test = np.empty((len(testing_cases),), dtype='int')
            
            for i, case_id in enumerate(training_cases):
                # Load the image file
                image_file = os.path.join(self.data, f'{case_id}.jpeg')
                image = Image.open(image_file)

                # Resize the image and convert it to the output format
                # image = image.resize(image_size)
                image = np.array(image, dtype='float32')
                image /= 255.0

                # Store the image and its label in the output data structure
                x_train[i] = image
            # Load and convert the images for testing
            for i, case_id in enumerate(testing_cases):
                # Load the image file
                image_file = os.path.join(self.data, f'{case_id}.jpeg')
                image = Image.open(image_file)

                # Resize the image and convert it to the output format
                # image = image.resize(image_size)
                image = np.array(image, dtype='float32')
                image /= 255.0

                # Store the image and its label in the output data structure
                x_test[i] = image
            self.training_data = x_train
            self.testing_data = x_test
        else:
            norm = preprocessing.MinMaxScaler()
            norm.fit(merged_data_train[self.vars])
            self.training_data = norm.transform(merged_data_train[self.vars])
            self.testing_data = norm.transform(merged_data_test[self.vars])

    

    def __str__(self) -> str:
        return self.name    

# define attention
def cross_modal_attention(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = tf.keras.layers.MultiHeadAttention(num_heads = 4,key_dim=50)(x, y)
    a2 = tf.keras.layers.MultiHeadAttention(num_heads = 4,key_dim=50)(y, x)
    a1 = a1[:,0,:]
    a2 = a2[:,0,:]
    return tf.keras.layers.concatenate([a1, a2])

def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = tf.keras.layers.MultiHeadAttention(num_heads = 4, key_dim=50)(x, x)
    attention = attention[:,0,:]
    return attention

def main(argv):
    parser = argparse.ArgumentParser(description='Create a multimodal model.')
    parser.add_argument('-m', '--modalities', nargs='+', help='a list of modalities')
    parser.add_argument('-f', '--filepaths', nargs='+', help='a list of filepaths')
    parser.add_argument('-o', '--output', help='an output directory')
    args = parser.parse_args(argv)
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    modality_dir = os.path.join(args.output, f"{len(args.modalities)}-modality")
    if not os.path.exists(modality_dir):
        os.mkdir(modality_dir)
    output = os.path.join(modality_dir, '+'.join(args.modalities))
    if not os.path.exists(output):
        os.mkdir(output)
    ### create logger
    logger_file = os.path.join(output, "logger.log")
    logging.basicConfig(filename=logger_file, filemode='w', encoding='utf-8', \
        level=logging.INFO)
    logging.info(f"Modalities {args.modalities}\nFilepaths {args.filepaths}\Output {args.output}")
    modality_dict = {}
    for m, f in zip(args.modalities, args.filepaths):
        modality_dict[m] = f
    training_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/training_cases.csv")["case_id"])
    testing_cases = list(pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/testing_cases.csv")["case_id"])
    clinical_df = pd.read_csv("/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_clinical_data.csv", index_col = 0)
    diagnosis = clinical_df[["vital_status_Dead", "case_id"]]
    def_hyperparameters = {
        "learning_rate" : 0.001,
        "n_epochs" : 40,
        "batch_size" : 30,
        "CNV" : {
            "dim_hidden":512,
            "num_hidden":3,
            "dim_end":1024,
            "feature_reduction":True
        },
        "EPIGENOMIC" : {
            "dim_hidden":512,
            "num_hidden":3,
            "dim_end":1024,
            "feature_reduction":True
        },
        "TRANSCRIPTOMIC" : {
            "dim_hidden":512,
            "num_hidden":3,
            "dim_end":1024,
            "feature_reduction":True
        },
        "CLINICAL" : {
            "dim_hidden":16,
            "num_hidden":2,
            "dim_end":32,
            "feature_reduction":False
        },
        "IMAGING" : {
            "dim_hidden":16,
            "num_hidden":2,
            "dim_end":32
        }}
    logging.info("Loading data...")
    mmm = MultimodalModel(modality_dict, settings= def_hyperparameters, attention = True)
    x_train, y_train, x_test, y_test = mmm.merge_data(y = diagnosis, train_ids = training_cases, test_ids = testing_cases, load_data = True, rebalance_data= True)
    logging.info("Creating model...")
    mmm.create_model()
    tf.keras.utils.plot_model(model = mmm.model, to_file = os.path.join(output, "architecture.png"), show_shapes=True, show_layer_names=True, show_layer_activations=True)
    logging.info(f"Training model for {def_hyperparameters['n_epochs']} epochs...")
    model, history = mmm.train_model(n_epochs=10)
    model.save(os.path.join(output, "mm_model"))
    logging.info("Testing model...")
    mmm.test_model(os.path.join(output, "testing"))

    mmm.generate_summary(os.path.join(output, "summary.txt"))
    plots.plot_convergence(history, os.path.join(output, "convergence"))

if __name__ == "__main__":
    main(sys.argv[1:])