import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import copy
import os
sns.set()
def plot_confusion(model, x_test, y_test, filepath = None):
	y_test_cat = tf.keras.utils.to_categorical(y_test)
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

def plot_convergence(hist, filepath):
	history = copy.deepcopy(hist)
	opt_curves = pd.DataFrame({'training_loss':history.history['loss'], 'val_loss':history.history['val_loss']})
	opts = sns.lineplot(opt_curves)
	opts.get_figure().savefig(filepath)
	plt.clf()