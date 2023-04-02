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


class MultimodalModel():
    def __init__(self, modalities = None) -> None:
        self.modalities: dict = {}
        if modalities is not None:
            if isinstance(modalities, list):
                for modality in modalities:
                    self.modalities[modality] = Modality(name = modality)
            elif isinstance(modalities, dict):
                for modality in modalities:
                    self.modalities[modality] = Modality(name = modality, data = modalities[modality])
            else:
                raise TypeError("Modalities should be passed as either a list of strings or a dictionary with modality names corresponding to data paths.")
    def merge_data(self, y, merge_on = "case_id", target_name = "vital_status_Dead", train_ids = None, test_ids = None):
        modalities = list(self.modalities.values())
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
            y_train = data_training[target_name].astype(int)
            y_test = data_testing[target_name].astype(int)
            return x_train, y_train, x_test, y_test
        else:
            return self.merged

    


class Modality():
    def __init__(self, name, data = None, omit_vars = ["CASE_ID", "index"]) -> None:
        self.name = name
        self.vars = None
        if data is not None:
            self.data: pd.DataFrame = pd.read_csv(data, index_col = 0)
            self.vars = set(self.data.columns) - set(omit_vars)
