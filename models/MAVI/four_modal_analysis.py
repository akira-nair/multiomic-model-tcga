import optuna
import unimodal_model as um
import datetime
import pandas as pd
import numpy as np
from optuna.visualization import plot_optimization_history, plot_contour, plot_parallel_coordinate
import plotly
import datetime
import sys
import os
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.merging import concatenate
from keras.models import Sequential
from keras.layers import Dense,Dropout, BatchNormalization, MultiHeadAttention
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging
sns.set()

def optimization(x_train, y_train, x_test, y_test, clinical_vars, cnv_vars, epigenomic_vars, transcriptomic_vars, trials):
    def objective(trial):
        os.mkdir(f"{dir}/{trial.number}")
        os.mkdir(f"{dir}/{trial.number}/models")
        # global model training
        n_epochs = trial.suggest_int("epochs", 20, 200, 30)
        lr = 1**(-1 * trial.suggest_int("lr", 1, 5))
        
        # Clinical hyperparameters
        clinical_final_dim = trial.suggest_int("clinical_final_dim", 8, 32, 4)

        # CNV hyperparameters
        cnv_rf_est = trial.suggest_int("cnv_dim_hidden", 50, 200, 25)
        cnv_n_hidden = trial.suggest_int("cnv_n_hidden", 1, 3)
        cnv_dim_hidden = trial.suggest_int("cnv_dim_hidden", 32, 512, 16)
        cnv_final_dim = trial.suggest_int("cnv_final_dim", 16, 128, 16)

        # Epigenomic hyperparameters
        epi_rf_est = trial.suggest_int("epi_dim_hidden", 50, 200, 25)
        epi_n_hidden = trial.suggest_int("epi_n_hidden", 1, 3)
        epi_dim_hidden = trial.suggest_int("epi_dim_hidden", 32, 512, 16)
        epi_final_dim = trial.suggest_int("epi_final_dim", 16, 128, 16)

        # Transcriptomic hyperparameters
        tsc_rf_est = trial.suggest_int("tsc_dim_hidden", 50, 200, 25)
        tsc_n_hidden = trial.suggest_int("tsc_n_hidden", 1, 3)
        tsc_dim_hidden = trial.suggest_int("tsc_dim_hidden", 32, 512, 16)
        tsc_final_dim = trial.suggest_int("tsc_final_dim", 16, 128, 16)

        # divide back into modalities
        X_train_clinical = x_train[clinical_vars]
        X_train_cnv = x_train[cnv_vars]
        X_train_epigenomic = x_train[epigenomic_vars]
        X_train_transcriptomic = x_train[transcriptomic_vars]

        # feature reduction for CNV
        sel_cnv = SelectFromModel(RandomForestClassifier(n_estimators = cnv_rf_est, random_state=42))
        sel_cnv.fit(X_train_cnv, y_train)
        selected_feat_cnv = pd.DataFrame(X_train_cnv).columns[(sel_cnv.get_support())]
        logging.info(f"{len(selected_feat_cnv)} features were selected for CNV data")
        X_cnv_encoded = X_train_cnv[selected_feat_cnv]

        # feature reduction for Epigenomic
        sel_epi = SelectFromModel(RandomForestClassifier(n_estimators = epi_rf_est, random_state=42))
        sel_epi.fit(X_train_epigenomic, y_train)
        selected_feat_epi = pd.DataFrame(X_train_epigenomic).columns[(sel_epi.get_support())]
        logging.info(f"{len(selected_feat_epi)} features were selected for EPIGENETIC data")
        X_epigenomic_encoded = X_train_epigenomic[selected_feat_epi]

        # feature reduction for Transcriptomic
        sel_tsc = SelectFromModel(RandomForestClassifier(n_estimators = tsc_rf_est, random_state=42))
        sel_tsc.fit(X_train_transcriptomic, y_train)
        selected_feat_tsc = pd.DataFrame(X_train_transcriptomic).columns[(sel_tsc.get_support())]
        logging.info(f"{len(selected_feat_tsc)} features were selected for TRANSCRIPTOMIC data")
        X_transcriptomic_encoded = X_train_transcriptomic[selected_feat_tsc]

        ## Construct model architecture

        # clinical branch
        clinical_input = Input(shape=(X_train_clinical.shape[1], ), name='clinical_input')
        clinical_branch = Dense(clinical_final_dim, input_dim=X_train_clinical.shape[1], name='clinical_branch')(clinical_input)

        # cnv branch
        cnv_input = Input(shape=(X_cnv_encoded.shape[1],), name='cnv_input')
        cnv_hidden = Dense(cnv_dim_hidden, input_dim=X_cnv_encoded.shape[1], name='cnv_hidden_0')(cnv_input)
        hidden_layers = [cnv_hidden]
        for n_layer in range(cnv_n_hidden - 1):
            hidden_layers.append(Dense(cnv_dim_hidden, input_dim=cnv_dim_hidden, name=f'cnv_hidden_{n_layer+1}')(hidden_layers[n_layer]))
        cnv_branch = Dense(cnv_final_dim, input_dim=cnv_dim_hidden, name='cnv_branch')(hidden_layers.pop())

        # epigenomic branch
        epi_input = Input(shape=(X_epigenomic_encoded.shape[1],), name='epigenomic_input')
        epi_hidden = Dense(epi_dim_hidden, input_dim=X_epigenomic_encoded.shape[1], name='epi_hidden_0')(epi_input)
        hidden_layers = [epi_hidden]
        for n_layer in range(epi_n_hidden - 1):
            hidden_layers.append(Dense(epi_dim_hidden, input_dim=epi_dim_hidden, name=f'epi_hidden_{n_layer+1}')(hidden_layers[n_layer]))
        epi_branch = Dense(epi_final_dim, input_dim=epi_dim_hidden, name='epi_branch')(hidden_layers.pop())

        # transcriptomic branch
        tsc_input = Input(shape=(X_transcriptomic_encoded.shape[1],), name='transcriptomic_input')
        tsc_hidden = Dense(tsc_dim_hidden, input_dim=X_transcriptomic_encoded.shape[1], name='tsc_hidden_0')(tsc_input)
        hidden_layers = [tsc_hidden]
        for n_layer in range(tsc_n_hidden - 1):
            hidden_layers.append(Dense(tsc_dim_hidden, input_dim=tsc_dim_hidden, name=f'tsc_hidden_{n_layer+1}')(hidden_layers[n_layer]))
        tsc_branch = Dense(tsc_final_dim, input_dim=tsc_dim_hidden, name='tsc_branch')(hidden_layers.pop())

        # No attention
        modalities_branches_na = [clinical_branch, cnv_branch, epi_branch, tsc_branch]
        concat_na = concatenate(modalities_branches_na)
        predictions_na = Dense(1, activation='sigmoid', name='main_output')(concat_na)
        
        # Self attention
        clinical_att = self_attention(clinical_branch)
        cnv_att = self_attention(cnv_branch)
        epi_att = self_attention(epi_branch)
        tsc_att = self_attention(tsc_branch)
        modalities_branches_sa = [clinical_att, cnv_att, epi_att, tsc_att]
        concat_sa = concatenate(modalities_branches_sa)
        predictions_sa = Dense(1, activation='sigmoid', name='main_output')(concat_sa)

        # Cross-modal attention
        clinical_cnv_cm_att = cross_modal_attention(clinical_branch, cnv_branch)
        clinical_epi_cm_att = cross_modal_attention(clinical_branch, epi_branch)
        clinical_tsc_cm_att = cross_modal_attention(clinical_branch, tsc_branch)
        cnv_epi_cm_att = cross_modal_attention(cnv_branch, epi_branch)
        cnv_tsc_cm_att = cross_modal_attention(cnv_branch, tsc_branch)
        epi_tsc_cm_att = cross_modal_attention(epi_branch, tsc_branch)
        modalities_branches_cma = [clinical_cnv_cm_att, clinical_epi_cm_att, clinical_tsc_cm_att, cnv_epi_cm_att, cnv_tsc_cm_att, epi_tsc_cm_att]
        concat_cma = concatenate(modalities_branches_cma)
        predictions_cma = Dense(1, activation='sigmoid', name='main_output')(concat_cma)

        # Self and cross-modal attention
        clinical_cnv_sacm_att = cross_modal_attention(clinical_att, cnv_att)
        clinical_epi_sacm_att = cross_modal_attention(clinical_att, epi_att)
        clinical_tsc_sacm_att = cross_modal_attention(clinical_att, tsc_att)
        cnv_epi_sacm_att = cross_modal_attention(cnv_att, epi_att)
        cnv_tsc_sacm_att = cross_modal_attention(cnv_att, tsc_att)
        epi_tsc_sacm_att = cross_modal_attention(epi_att, tsc_att)
        modalities_branches_sacma = [clinical_cnv_sacm_att, clinical_epi_sacm_att, clinical_tsc_sacm_att, cnv_epi_sacm_att, cnv_tsc_sacm_att, epi_tsc_sacm_att]
        concat_sacma = concatenate(modalities_branches_sacma)
        predictions_sacma = Dense(1, activation='sigmoid', name='main_output')(concat_sacma)

        # Create model
        model_na = Model(inputs=[clinical_input, cnv_input, epi_input, tsc_input], outputs=predictions_na)
        model_sa = Model(inputs=[clinical_input, cnv_input, epi_input, tsc_input], outputs=predictions_sa)
        model_cma = Model(inputs=[clinical_input, cnv_input, epi_input, tsc_input], outputs=predictions_cma)
        model_sacma = Model(inputs=[clinical_input, cnv_input, epi_input, tsc_input], outputs=predictions_sacma)
        models = [model_na, model_sa, model_cma, model_sacma]
        trial_testing_accs = []
        trial_validation_accs = []
        model_names = {
            model_na:"no_attention",
            model_sa:"self_attention",
            model_cma:"cross_modal_attention",
            model_sacma:"self_and_cross_modal_attention"
        }
        for model in models:
            reset_weights(model)
            model_name = model_names[model]
            # Save model architecture
            tf.keras.utils.plot_model(model, to_file=f"{dir}/{trial.number}/models/architecture_{model_name}.png",show_shapes=True, show_layer_names=True, show_layer_activations=True)
            # Clear session
            tf.keras.backend.clear_session()
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
            model.compile(optimizer = optimizer, loss = tf.keras.losses.BinaryCrossentropy(), metrics = tf.keras.metrics.BinaryAccuracy())
            # Train model
            history = model.fit([X_train_clinical, X_cnv_encoded, X_epigenomic_encoded, X_transcriptomic_encoded], y_train, epochs=n_epochs, validation_split=0.1, verbose=2)

            # Save model
            model.save(f"{dir}/{trial.number}/models/4m_model_{model_name}")

            # Plot optimization results
            plt.rcParams['figure.dpi'] = 400
            sns.set_theme(style="white", palette=None)
            fig, (ax1, ax2) = plt.subplots(nrows = 2)
            ax1.plot(history.history['binary_accuracy'])
            ax1.plot(history.history['val_binary_accuracy'])
            ax2.plot(history.history['loss'])
            ax2.plot(history.history['val_loss'])
            ax1.set_title('accuracy')
            ax2.set_title('loss')
            ax1.set_ylabel('accuracy')
            ax2.set_ylabel('loss')
            plt.xlabel('epoch')
            fig.legend(['train', 'validation'], loc='upper right')
            fig.tight_layout() 
            plt.savefig(f"{dir}/{trial.number}/optimization_plot_{model_name}.png")
            plt.close()

            X_test_clinical = x_test[clinical_vars]
            X_test_cnv = x_test[cnv_vars]
            X_test_epigenomic = x_test[epigenomic_vars]
            X_test_transcriptomic = x_test[transcriptomic_vars]
            X_test_encoded = [X_test_clinical, X_test_cnv[selected_feat_cnv], \
                    X_test_epigenomic[selected_feat_epi], X_test_transcriptomic[selected_feat_tsc]]

            accuracy = um.plot_confusion(model = model,
                    x_test = X_test_encoded,
                    y_test = y_test,
                    filepath = f"{dir}/{trial.number}/testing_results_{model_name}.png")
            trial_testing_accs.append(accuracy)
            trial_validation_accs.append(history.history["val_binary_accuracy"].pop())

        validation_accuracies.append(trial_validation_accs)
        testing_accuracies.append(trial_testing_accs)
        # Optimize on no attention model
        return validation_accuracies[0]

        # return history.history["val_binary_accuracy"].pop()
    # Run the optimization
    os.chdir("/users/anair27/data/anair27/singh-lab-TCGA-project/hyperparameter-tuning")
    timestamp = datetime.datetime.now().strftime("%h-%d-%H:%M:%S")
    dir = f"four-modal-analysis_{timestamp}"
    validation_accuracies = [["No attention", "Self attention", \
        "Cross-modal attention", "Self and Cross-modal attention"]]
    testing_accuracies = [["No attention", "Self attention", \
        "Cross-modal attention", "Self and Cross-modal attention"]]
    os.mkdir(f"{dir}")
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials=trials)
    testing_accuracies_df = pd.DataFrame(testing_accuracies)
    validation_accuracies_df = pd.DataFrame(validation_accuracies)
    testing_accuracies_df.to_csv(f"{dir}/testing_accuracies.csv")
    validation_accuracies_df.to_csv(f"{dir}/validations_accuracies.csv")
    logging.info('--------------------------------')
    logging.info("Hyperparameter optimization results:")
    logging.info(f"BEST PARAMETERS {study.best_params} HAD BEST VALUE {study.best_value}")
    results: pd.DataFrame = study.trials_dataframe()
    results.to_csv(f"{dir}/results.csv")

def reset_weights(model):
    import keras.backend as K
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel.initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias.initializer'):
            layer.bias.initializer.run(session=session) 

def read_data():
    os.chdir("/users/anair27/data/anair27/singh-lab-TCGA-project/MADDi_model/")
    # Load clinical data
    clinical_data = pd.read_csv("./data_processed/clinical_data.csv", index_col = 0).reset_index()
    # Load CNV data
    all_cnv_data = pd.read_csv("./data_processed/cnv_data.csv", index_col = 0)
    # Load epigenomic data
    all_epigenomic_data = pd.read_csv("./data_processed/epigenomic_data.csv", index_col = 0)
    # Load transcriptomic data
    all_transcriptomic_data = pd.read_csv("./data_processed/transcriptomic_data.csv", index_col = 0)
    clinical_data = clinical_data.set_index('case_id').reset_index()
    cnv_data = all_cnv_data.reset_index().rename(columns={"CASE_ID":"case_id"})
    epigenomic_data = all_epigenomic_data
    transcriptomic_data = all_transcriptomic_data.reset_index().drop(["index"], axis= 1)
    clinical_vars = clinical_data.columns
    cnv_vars = cnv_data.columns
    epigenomic_vars = epigenomic_data.columns
    transcriptomic_vars = transcriptomic_data.columns
    clinical_vars = list(set(clinical_vars) - set(["vital_status_Dead", "case_id", "index"]))
    cnv_vars = list(set(cnv_vars) - set(["case_id"]))
    epigenomic_vars = list(set(epigenomic_vars) - set(["case_id"]))
    transcriptomic_vars = list(set(transcriptomic_vars) - set(["case_id"]))
    return clinical_data, cnv_data, epigenomic_data, transcriptomic_data, clinical_vars, cnv_vars, epigenomic_vars, transcriptomic_vars

# define attention
def cross_modal_attention(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = MultiHeadAttention(num_heads = 4,key_dim=50)(x, y)
    a2 = MultiHeadAttention(num_heads = 4,key_dim=50)(y, x)
    a1 = a1[:,0,:]
    a2 = a2[:,0,:]
    return concatenate([a1, a2])

def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = MultiHeadAttention(num_heads = 4, key_dim=50)(x, x)
    attention = attention[:,0,:]
    return attention

def main(args):
    os.chdir("/users/anair27/data/anair27/singh-lab-TCGA-project/hyperparameter-tuning")
    logging.basicConfig(filename='analysis_four_modal.log', filemode='w', encoding='utf-8', \
        level=logging.INFO)
    logging.info("\n\n-----\n Beginning an analysis of four modalities...\n-----\n\n")
    clinical_data, cnv_data, epigenomic_data, transcriptomic_data, clinical_vars, cnv_vars, epigenomic_vars, transcriptomic_vars = read_data()
    logging.info("Data has been read. Now, the 4 modalities will be merged.")
    merged_data = cnv_data.merge\
    (clinical_data, on = "case_id").drop_duplicates().merge(\
            epigenomic_data, on = "case_id").drop_duplicates().merge(\
                transcriptomic_data, on = "case_id").drop_duplicates()
    logging.info("Merged dataset has been created. Now, an ADASYN class balancing strategy will be used.")
    cols = list(set(merged_data.columns) - set(["vital_status_Dead", "case_id", "index"]))
    X = merged_data[cols].astype(float)
    Y = merged_data["vital_status_Dead"].astype(int)
    adasyn = ADASYN(random_state=42)
    x_rs, y_rs = adasyn.fit_resample(X, Y)
    logging.info('Original dataset shape had: ', Counter(Y))
    logging.info('Resample dataset shape had: ', Counter(y_rs))
    # split data
    X_train, X_test, Y_train, Y_test = train_test_split(x_rs, y_rs, test_size=0.1, random_state=42)
    logging.info("Data has been split into training and testing subsets.")
    logging.info("\n\n Optuna will now optimize a model. The following parameters will be optimized for:")
    logging.info("Random forest feature reduction: n_estimators for CNV, epigenomic, and transcriptomic modalities")
    logging.info("Number of dimensions")
    optimization(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, clinical_vars=clinical_vars, cnv_vars=cnv_vars, epigenomic_vars=epigenomic_vars, transcriptomic_vars=transcriptomic_vars, trials=(int)(args[0]))
    logging.info("\n The four modal model optimization has completed.")

if __name__ == "__main__":
    main(sys.argv[1:])  
    