import tensorflow as tf
from tensorflow import keras 
from keras import optimizers
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
#Plotting a random image 
#====================================================================================#
#plt.imshow(train_images[90],cmap = plt.cm.binary )
#plt.show()
#====================================================================================#

#====================================================================================#
#Normalising the data 
#====================================================================================#
	#The data that keras loaded is in form of a numpy array
test_images = test_images/255.0
train_images = train_images/255.0
#====================================================================================#


#====================================================================================#
#Defining the model 
	#Sequential Model is like the horizontal stacked layer
#====================================================================================#
model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28,28)),
	keras.layers.Dense(512, activation = "relu" ),
	keras.layers.Dense(512, activation = "relu" ),
	keras.layers.Dense(256, activation = "relu" ),
	keras.layers.Dense(128, activation = "relu" ),
	keras.layers.Dense(10, activation = "softmax" )
	])
# Insert Hyperparameters
learning_rate = 0.01
training_epochs = 20
batch_size = 100
sgd = optimizers.SGD(lr=learning_rate)

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

#====================================================================================#



#====================================================================================#
#Training the model 
#====================================================================================#
model.fit(train_images, train_labels, epochs = 5, batch_size = batch_size,verbose = 2)
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
prediction = model.predict(test_images)

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i],cmap = plt.cm.binary)
	plt.xlabel("Actual : " + str(test_labels[i]))
	plt.title("Prediction : " + str(np.argmax(prediction[i])))
	plt.show()
#====================================================================================#


#====================================================================================#
# Saving the model 
#====================================================================================#
model.save("model.h5")
#====================================================================================#








