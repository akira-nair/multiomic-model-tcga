import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization, Flatten, Reshape
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
import seaborn as sns

def create_model(x_train, y_train, train: bool = True, n_epochs = 50, \
                 batch_size = None, lr = 0.001, n_hidden = 1, dim_hidden = 64, 
                 optimizer = 'sgd', validation_set = 0.1, activation = 'sigmoid'):
    model = Sequential()
    model.add(Dense(128, input_shape = (x_train.shape[1],), activation = "relu"))
    model.add(BatchNormalization())
    for layer in range(n_hidden):
        model.add(Dense(dim_hidden, activation = "relu"))
        model.add(BatchNormalization())
    model.add(Dense(int(dim_hidden / 4), activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(1, activation = activation))
    model.compile(optimizer = optimizer, \
                  loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
    if train:
        history = model.fit(x_train, y_train,  epochs=n_epochs, \
                            validation_split=validation_set, batch_size=batch_size,\
                            verbose=1)
        return model, history
    else:
        return model, None

def create_ae(x_train, latent_dim = 128, epochs= 10, lr=0.001):
    class Autoencoder(Model):
      def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = Sequential([
          Dense(latent_dim, activation='relu'),
        ])
        self.decoder = Sequential([
          Dense(x_train.shape[1], activation='sigmoid')
          # Reshape((28, 28))
        ])

      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    autoencoder = Autoencoder(latent_dim)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    autoencoder.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    autoencoder.fit(tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_train),
                epochs=epochs,
                shuffle=True)
    return autoencoder

def encode_input(autoencoder, x):
    return pd.DataFrame(autoencoder.encoder(tf.convert_to_tensor(x)).numpy())

def plot_confusion(model, x_test, y_test, filepath = None):
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)
    plt.close()
    sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred)), annot= True, cmap = "crest")
    plt.title(f"Accuracy: {round(score[1], 3)}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if filepath is not None:
      plt.savefig(filepath)
    plt.close()
    return round(score[1], 3)

def test_model(model, x_test, y_test):
    acc = []
    f1 = []
    precision = []
    recall = []
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    test_predictions = model.predict(x_test)
    test_label = to_categorical(y_test,2)
    true_label= np.argmax(test_label, axis =1)
    predicted_label= np.argmax(test_predictions, axis =1)
    cr = classification_report(true_label, predicted_label, output_dict=True)
    return cr