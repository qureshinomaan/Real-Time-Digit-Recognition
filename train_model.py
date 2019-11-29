import tensorflow as tf
from tensorflow import keras 
from keras.regularizers import l2
from keras import optimizers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt 
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

#====================================================================================#
#Loading the data set 
#====================================================================================#
data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
#====================================================================================#

#====================================================================================#
# Making the data for conv2d layer. 
#====================================================================================#
print(train_images.shape)
if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
#====================================================================================#



#====================================================================================#
#Plotting a ranpdom image 
#====================================================================================#
# plt.imshow(train_images[90],cmap = plt.cm.binary )
# plt.show()
# print(train_images[0]/255.0)
#====================================================================================#

#====================================================================================#
#Normalising the data 
#====================================================================================#
#The data that keras loaded is in form of a numpy array
test_images = test_images/255.0
print(train_images.shape)
train_images = train_images/255.0
#====================================================================================#


#====================================================================================#
#Defining the model 
	#Sequential Model is like the horizontal stacked layer
#====================================================================================#
model = keras.Sequential([
	keras.layers.Conv2D(64, 3, activation="relu", input_shape = (28,28,1)),
	keras.layers.MaxPooling2D(2),
	keras.layers.Conv2D(64, 3, activation="relu"),
	keras.layers.Dropout(0.5),
	keras.layers.MaxPooling2D(2),
	keras.layers.Conv2D(64, 3, activation="relu"),
	keras.layers.Flatten(),
	keras.layers.Dense(512, activation = "relu"),
	keras.layers.Dropout(0.75),
	keras.layers.Dense(256, activation = "relu"),
	keras.layers.Dropout(0.75),
	keras.layers.BatchNormalization(axis=-1, epsilon=0.001),
	keras.layers.Dense(512, activation = "relu"),
	keras.layers.Dropout(0.75),
	keras.layers.BatchNormalization(axis=-1, epsilon=0.001),
	keras.layers.Dense(128, activation = "relu"),
	keras.layers.Dense(10, activation = "softmax" )
	])
# Insert Hyperparameters
learning_rate = 0.01
training_epochs = 4
batch_size = 100
sgd = optimizers.SGD(lr=learning_rate)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

#====================================================================================#



#====================================================================================#
#Training the model 
#====================================================================================#
model.fit(train_images, train_labels, epochs = training_epochs, batch_size = batch_size)
#epochs is how many time you see an image 
#The images are randomly feed to the neural network because the way the images are fed to the 
#model tweaks the weights.
#====================================================================================#



#====================================================================================#
#Evaluating model against the test data getting the accuracy 
#====================================================================================#
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Accuracy of model is : ", test_acc)
#====================================================================================#


#====================================================================================#
#Predicting single elements 
#====================================================================================#
# prediction = model.predict(test_images)

# for i in range(5):
# 	plt.grid(False)
# 	plt.imshow(test_images[i],cmap = plt.cm.binary)
# 	plt.xlabel("Actual : " + str(test_labels[i]))
# 	plt.title("Prediction : " + str(np.argmax(prediction[i])))
# 	plt.show()
#====================================================================================#


#====================================================================================#
# Saving the model 
#====================================================================================#
model.save("model.h5")
#====================================================================================#








