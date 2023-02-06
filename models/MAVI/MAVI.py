#!/usr/bin/env python
'''
File        :   MAVI.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Multimodal attention and variational inference
'''
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from sklearn.preprocessing import normalize
import seaborn as sns
import umap
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class VAE():
    def __init__(self, input_dim, hidden_dim=512, latent_dim=32, activation='sigmoid', lr = 0.0001) -> None:
        # encoder
        input_layer = Input(shape = (input_dim,), name = 'encoder_input')
        hidden_1 = Dense(hidden_dim, activation = activation, name = 'hidden_layer_1')(input_layer)
        # define the probability distribution parameters of the latent space
        self.z_mean = Dense(latent_dim, name = 'latent_mean')(hidden_1)
        self.z_var = Dense(latent_dim, name = 'latent_log_var')(hidden_1)
        z = Lambda(self.sample_z, output_shape=(latent_dim, ), name='z')([self.z_mean, self.z_var])
        # decoder
        decoder_hidden = Dense(512, activation = activation, name = 'hidden_layer_2')
        reconstructed_input = Dense(input_dim, activation = None, name = 'reconstructed_input')
        # specifying a copy for the VAE as a whole
        h_p = decoder_hidden(z)
        outputs = reconstructed_input(h_p)
        # create model
        self.vae = Model(input_layer, outputs)
        self.encoder = Model(input_layer, [self.z_mean, self.z_var, z], name = 'encoder')
        # specifying a copy for the decoder (cannot use z as an input layer)
        d_in = Input(shape=(latent_dim,))
        d_h = decoder_hidden(d_in)
        d_out = reconstructed_input(d_h)
        self.decoder = Model(d_in, d_out, name = 'decoder')
        # define methods
        def vae_loss(y_true, y_pred):
            recon = K.sum(keras.metrics.mean_squared_error(y_true, y_pred), axis=-1)
            kl = 0.5 * K.sum(K.exp(self.z_var) + K.square(self.z_mean) - 1. - self.z_var, axis=-1)
            return recon + kl
    
        def KL_loss(y_true, y_pred):
            return(0.5 * K.sum(K.exp(self.z_var) + K.square(self.z_mean) - 1. - self.z_var, axis=1))

        def recon_loss(y_true, y_pred):
            return K.sum(keras.metrics.mean_squared_error(y_true, y_pred), axis=-1)
        # compile model
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        (self.vae).compile(optimizer=optimizer, loss=vae_loss, metrics = [KL_loss, recon_loss])
        # encoder = Model(input_layer, [z_mean, z_var, z], name = 'encoder')
        # decoder = Model(decoder_input, reconstructed_input, name = 'decoder')
    
    def sample_z(self, args):
        z_mean, z_var = args
        eps = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
        return z_mean + K.exp(z_var / 2) * eps
    
    

    def train(self, x_train, n_epochs = 10):
        self.history = self.vae.fit(x_train, x_train, epochs = n_epochs, validation_split=0.1, verbose = 2)
    
    def plot_latent_space(self, data_x, data_y, output= None):
        mu, var, _ = self.encoder.predict(data_x)
        reducer = umap.UMAP()
        embedding_mean = reducer.fit_transform(mu)
        embedding_var = reducer.fit_transform(var)
        data = pd.DataFrame({
            'Mean, UMAP 1':embedding_mean[:, 0], 
            'Mean, UMAP 2':embedding_mean[:, 1],
            'Variance, UMAP 1':embedding_var[:, 0], 
            'Variance, UMAP 2':embedding_var[:, 1],
            'vital status':data_y})
        sns.set(rc={'figure.figsize':(16,8), 'figure.dpi': 300})    
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.scatterplot(data=data, x = 'Mean, UMAP 1', y = 'Mean, UMAP 2', hue='vital status', ax=ax1)
        sns.scatterplot(data=data, x = 'Variance, UMAP 1', y = 'Variance, UMAP 2', hue='vital status', ax=ax2)
        fig.tight_layout()
        fig.show()
        if output is not None:
            fig.savefig(output)
    def get_embedding(self, data_x):
        return self.encoder.predict(data_x)[0]
        

'''
class MyLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        
        # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
        recon_loss = keras.metrics.mean_squared_error(x, z_decoded)
        print(recon_loss)
#         recon_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_var - K.square(z_mean) - K.exp(z_var), axis=-1)
#         return recon_loss + kl_loss
        return K.mean(recon_loss + kl_loss)
    
    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
def KL_loss(x, z_decoded):
    x = K.flatten(x)
    z_decoded = K.flatten(z_decoded)
    return-5e-4 * K.mean(1 + z_var - K.square(z_mean) - K.exp(z_var), axis=-1)

def recon_loss(x, z_decoded):
    x = K.flatten(x)
    z_decoded = K.flatten(z_decoded)
    return keras.metrics.mean_squared_error(x, z_decoded)
reducer = umap.UMAP()

embedding_mean = reducer.fit_transform(mu)
embedding_var = reducer.fit_transform(var)
data = pd.DataFrame({
    'Mean, UMAP 1':embedding_mean[:, 0], 
    'Mean, UMAP 2':embedding_mean[:, 1],
    'Variance, UMAP 1':embedding_var[:, 0], 
    'Variance, UMAP 2':embedding_var[:, 1],
    'vital status':y_train})
    sns.set(rc={'figure.figsize':(16,8), 'figure.dpi': 300})
sns.set_theme(style="whitegrid", palette="deep")
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.scatterplot(data=data, x = 'Mean, UMAP 1', y = 'Mean, UMAP 2', hue='vital status', ax=ax1)
sns.scatterplot(data=data, x = 'Variance, UMAP 1', y = 'Variance, UMAP 2', hue='vital status', ax=ax2)
fig.tight_layout()
'''