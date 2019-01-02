# Machine Learning Homework 4 - Image Classification

__author__ = '**'

# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import os
import sys
import pandas as pd

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier


epoch = 10

### Already implemented
def get_data(datafile):
	dataframe = pd.read_csv(datafile)
	dataframe = shuffle(dataframe)
	data = list(dataframe.values)
	labels, images = [], []
	for line in data:
		labels.append(line[0])
		images.append(line[1:])
	labels = np.array(labels)
	images = np.array(images).astype('float32')
	images /= 255
	return images, labels


### Already implemented
def visualize_weights(trained_model, num_to_display=20, save=True, hot=True):
	layer1 = trained_model.layers[0]
	weights = layer1.get_weights()[0]
	
	# Feel free to change the color scheme
	colors = 'hot' if hot else 'binary'
	try:
		os.mkdir('weight_visualizations')
	except FileExistsError:
		pass
	for i in range(num_to_display):
		wi = weights[:,i].reshape(28, 28)
		plt.imshow(wi, cmap=colors, interpolation='nearest')
		if save:
			plt.savefig('./weight_visualizations/unit' + str(i) + '_weights.png')
		else:
			plt.show()


### Already implemented
def output_predictions(predictions):
	with open('predictions.txt', 'w+') as f:
		for pred in predictions:
			f.write(str(pred) + '\n')


def plot_history(history):
	train_loss_history = history.history['loss']
	val_loss_history = history.history['val_loss']

	train_acc_history = history.history['acc']
	val_acc_history = history.history['val_acc']

	val_loss_history = history.history['val_loss']
	val_loss_history = np.asarray(val_loss_history)

	epochs_as_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	epochs_as_array = np.asarray(epochs_as_array)
	plt.plot(train_loss_history, epochs_as_array, "blue")
	plt.plot(val_loss_history, epochs_as_array, "red")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.show()

	train_acc_history = history.history['acc']
	train_acc_history = np.asarray(train_acc_history)
	val_acc_history = history.history['val_acc']
	val_acc_history = np.asarray(val_acc_history)

	plt.plot(train_acc_history, epochs_as_array, "blue")
	plt.plot(val_acc_history, epochs_as_array, "red")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.show()
	# plot



def create_mlp(args=None):
	# You can use args to pass parameter values to this method

	# Define model architecture
	model = Sequential()
	model.add(Dense(units=64, activation='relu', input_dim=28*28))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(units = 10, activation = 'softmax'))
	# Define Optimizer
	optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	# Compile
	model.compile(loss= 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model

def train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=None):
	# You can use args to pass parameter values to this method
	y_train = keras.utils.to_categorical(y_train, num_classes=10)
	model = create_mlp(args)
	history = model.fit(x_train, y_train, epochs=epoch, validation_split=0.1, batch_size=100)
	return model, history

def create_cnn(args=None):
	# You can use args to pass parameter values to this method

	# 28x28 images with 1 color channel
	input_shape = (28, 28, 1)

	# Define model architecture
	model = Sequential()
	model.add(Conv2D(32, kernel_size = (3, 3), strides = 1, activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2,2), strides = 1))
	# can add more layers here...
	model.add(Flatten())
	
	# can add more layers here...
	model.add(Dense(units=10, activation='softmax'))
	
	# Optimizer
	optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.8, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	# Compile
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model


def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
	# You can use args to pass parameter values to this method
	x_train = x_train.reshape(-1, 28, 28, 1)
	y_train = keras.utils.to_categorical(y_train, num_classes=10)
	model = create_cnn(args)
	history = model.fit(x_train, y_train, epochs=epoch, validation_split=0.1, batch_size=6000)
	return model, history


def train_and_select_model(train_csv, model_type=''):
	"""Optional method. You can write code here to perform a 
	parameter search, cross-validation, etc. """

	x_train, y_train = get_data(train_csv)
	
	args = {
		'learning_rate': 0.01,
	}
	if model_type == 'cnn':
		model, history = train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=args)
	else:
		model, history = train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=args)
	#validation_accuracy = history.history['val_acc']
	
	return model, history


if __name__ == '__main__':
	### Before you submit, switch this to grading_mode = False and rerun ###
	grading_mode = False
	if grading_mode:
		# When we grade, we'll provide the file names as command-line arguments
		if (len(sys.argv) != 3):
			print("Usage:\n\tpython3 fashion.py train_file test_file")
			exit()
		train_file, test_file = sys.argv[1], sys.argv[2]
		"""
		# train your best model
		best_model =
		
		# use your best model to generate predictions for the test_file
		predictions = []
		output_predictions(predictions)
		"""
		# Include all of the required figures in your report. Don't generate them here.

	else:
		train_file = 'fashion_train.csv'
		test_file = 'fashion_test.csv'
		# MLP
		
		mlp_model, mlp_history = train_and_select_model(train_file)
		print(mlp_model.summary())
		visualize_weights(mlp_model, 20)
		plot_history(mlp_history)
		
		
		# CNN
		
		cnn_model, cnn_history = train_and_select_model(train_file, 'cnn')
		print(cnn_model.summary())
		plot_history(cnn_history)
	
