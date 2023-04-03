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
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import Dense,Dropout,BatchNormalization, Flatten, Reshape, Conv2D, MaxPooling2D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

class MultimodalModel():
    """
    Multimodal model
    """
    def __init__(self, modalities = None) -> None:
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
        self.model = None
    
    def add_modality(self, name, data):
        new_modality = Modality(name, data)
        self.modalities.append(new_modality)

    def merge_data(self, y, merge_on = "case_id", target_name = "vital_status_Dead", train_ids = None, test_ids = None, load_data = False):
        modalities = []
        for modality in self.modalities:
            if modality.is_imaging == False:
                modalities.append(modality)
        if len(modalities) == 1:
            self.merged = modalities[0].data
        else:
            print(f"Merging {modalities[0].name} and {modalities[1].name}...")
            self.merged = modalities[0].data.merge(modalities[1].data, on = merge_on).drop_duplicates()
            if len(modalities) > 2:
                for modality in modalities[2:]:
                    print(f"Merging in {modality.name}...")
                    self.merged = self.merged.merge(modality.data, on = merge_on).drop_duplicates()
        print(f"Merging target variable data (y)...")
        self.merged = self.merged.merge(y, on = merge_on).drop_duplicates()
        if train_ids is not None and test_ids is not None:
            data_training = self.merged[self.merged[merge_on].isin(train_ids)]
            data_testing = self.merged[self.merged[merge_on].isin(test_ids)]
            x_train = data_training.loc[:, self.merged.columns != "vital_status_dead"]
            x_test = data_testing.loc[:, self.merged.columns != target_name]
            self.y_train = data_training[target_name].astype(int)
            self.y_test = data_testing[target_name].astype(int)
            if load_data:
                for modality in self.modalities:
                    modality.load_modality_data(x_train, x_test)
            return x_train, self.y_train, x_test, self.y_test
        else:
            return self.merged

    def create_model(self):
        model_inputs_layers = []
        modality_end_layers = []
        for modality in self.modalities:
            if modality.is_imaging:
                input = tf.keras.layers.Input(shape = modality.image_size, name=f"{modality.name}_input_layer")
                conv1 = tf.keras.layers.Conv2D(64, (3, 3),  activation='relu')(input)
                pool1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)
                flatten = tf.keras.layers.Flatten()(pool1)
                modality_end = tf.keras.layers.Dense(256, activation='relu', name=f"{modality.name}_end")(flatten)
            else:
                # define input layer
                input = tf.keras.layers.Input(shape=(modality.training_data.shape[1], ), name=f"{modality.name}_input_layer")
                modality_end = tf.keras.layers.Dense(256, activation='relu', name=f"{modality.name}_end")(input)
            model_inputs_layers.append(input)
            modality_end_layers.append(modality_end)
        concat = tf.keras.layers.concatenate(modality_end_layers)
        predictions = tf.keras.layers.Dense(2, activation='sigmoid', name='output')(concat)
        self.model: tf.keras.models.Model = tf.keras.models.Model(inputs=model_inputs_layers, outputs=predictions)
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        self.model.compile(optimizer = optimizer, loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])

    def train_model(self, n_epochs = 10):
        if self.model is None:
            raise AttributeError("Model has not been instantiated. Call create_model() first.")
        training_inputs = [m.training_data for m in self.modalities]    
        self.history: tf.keras.callbacks.History = self.model.fit(training_inputs, tf.keras.utils.to_categorical(self.y_train), epochs=n_epochs, validation_split=0.1, verbose=2)
        return self.model, self.history.history 
    
    def test_model(self, output):
        testing_inputs = [m.testing_data for m in self.modalities]
        plots.plot_confusion(self.model, testing_inputs, self.y_test, output)


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
            self.training_data = merged_data_train[self.vars]
            self.testing_data = merged_data_test[self.vars]

    def __str__(self) -> str:
        return self.name    
