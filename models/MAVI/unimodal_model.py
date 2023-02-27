import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization, Flatten, Reshape, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
import seaborn as sns

def create_model(x_train, y_train, train: bool = True, n_epochs = 50, \
                 batch_size = None, lr = 0.001, n_hidden = 1, dim_hidden = 64, 
                 optimizer = 'sgd', validation_set = 0.1, activation = 'sigmoid'):
	y_train = to_categorical(y_train)
	model = Sequential()
	model.add(Dense(128, input_shape = (x_train.shape[1],), activation = "relu"))
	model.add(BatchNormalization())
	for layer in range(n_hidden):
		model.add(Dense(dim_hidden, activation = "relu"))
		model.add(BatchNormalization())
	model.add(Dense(int(dim_hidden / 4), activation = "relu"))
	model.add(BatchNormalization())
	model.add(Dense(2, activation = activation))
	model.compile(optimizer = optimizer, \
					loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
	if train:
		history = model.fit(x_train, y_train,  epochs=n_epochs, \
							validation_split=validation_set, batch_size=batch_size,\
							verbose=1)
		return model, history
	else:
		return model, None

def create_model_image(x_train, y_train, image_shape, train: bool = True, n_epochs = 50, \
                 batch_size = None, lr = 0.001, n_hidden = 1, dim_hidden = 64, 
                 optimizer = 'sgd', validation_set = 0.1, activation = 'sigmoid'):

	class_weights = compute_class_weight('balanced', classes=[0, 1], y=list(y_train))
	class_weights = {0: class_weights[0], 1: class_weights[1]}
	y_train = to_categorical(y_train)
	model = Sequential()
	model.add(Conv2D(dim_hidden, (3, 3),  activation='relu', input_shape=image_shape))
	model.add(MaxPooling2D((2, 2)))
	# model.add(Dropout(0.1))
	for layer in range(n_hidden - 1):
		model.add(Conv2D(dim_hidden, (3, 3), activation='relu'))
		model.add(MaxPooling2D((2, 2)))
		# model.add(Dropout(0.1))
	model.add(Flatten())
	model.add(Dense(2, activation = activation))
	model.compile(optimizer = optimizer, \
					loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
	if train:
		history = model.fit(x_train, y_train,  epochs=n_epochs, \
							validation_split=validation_set, batch_size=batch_size,\
							verbose=1, class_weight = class_weights)
		return model, history
	else:
		return model, None

def verify_images(image_directory):
	"""verifies that the image directory has images of appropriate dimensions"""
	img_files = os.listdir(image_directory)
	sizes = set()
	for img in img_files:
		if img[-5:] == '.jpeg':
			im = Image.open(os.path.join(image_directory, img))
			sizes.add(im.size)
	if len(sizes) != 1:
		raise(f"Dimensions of images in directory are inconsistent, found sizes: {sizes}")
	else:
		print("Images are consistent.")

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
    autoencoder.fit(x_train, x_train,
                epochs=epochs,
                shuffle=True)
    return autoencoder

def encode_input(autoencoder, x):
    return autoencoder.encoder.predict(x)

def plot_confusion(model, x_test, y_test, filepath = None):
	y_test_cat = to_categorical(y_test)
	score = model.evaluate(x_test, y_test_cat, verbose=0)
	print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
	y_pred = model.predict(x_test)
	y_pred = np.argmax(y_pred, axis=1)
	plt.close()
	sns.heatmap(pd.DataFrame(confusion_matrix(y_test, y_pred)), annot= True, cmap = "crest")
	plt.title(f"Accuracy: {round(score[1], 3)}")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	if filepath is not None:
		plt.savefig(filepath)
	else:
		plt.show()

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