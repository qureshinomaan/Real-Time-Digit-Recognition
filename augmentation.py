#Augmentation code Taken from here 
# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/

# Random Rotations
from tensorflow import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# define data preparation
datagen = ImageDataGenerator(rotation_range=60)
shift = 0.5
datagensh = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
# fit parameters from data
datagen.fit(X_train)
datagensh.fit(X_train)



def get_rotated():
	datagen = ImageDataGenerator(rotation_range=60)
	X_rotated, Y_rotated = None, None
	for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=y_train.shape[0]//10):
		# create a grid of 3x3 images
		X_rotated = X_batch
		Y_rotated = y_batch 
		X_rotated = X_rotated.reshape(y_train.shape[0]//10, 28, 28)
		print(X_rotated.shape, Y_rotated.shape)
		print("In loop ")
		break
	return X_rotated, Y_rotated

def get_shifted(): 
	X_shifted, Y_shifted = None, None
	for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=y_train.shape[0]//10):
		# create a grid of 3x3 images
		X_shifted = X_batch
		Y_shifted = y_batch 
		X_shifted = X_shifted.reshape(y_train.shape[0]//10, 28, 28)
		print(X_shifted.shape)
		print("In loop ")
		break
	return X_shifted, Y_shifted

