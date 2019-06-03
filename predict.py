import tensorflow as tf
import random
import cv2
from getting_img import get_img
from tensorflow import keras 
from skimage import color
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt 
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context


#====================================================================================#
# Test data importing from keras#
#====================================================================================#
data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
test_images = test_images/255.0
train_images = train_images/255.0
#====================================================================================#


#====================================================================================#
# Loading the trained model. #
#====================================================================================#
model = keras.models.load_model("model.h5")
#====================================================================================#


#====================================================================================#
#This function selects a random image from the test data. #
#====================================================================================#
def rands():
	rand = random.randint(0,10001)
	prediction = model.predict(test_images[rand][np.newaxis,:,:])
	plt.grid(False)
	print(test_images[rand])
	plt.imshow(test_images[rand],cmap = plt.cm.binary)
	plt.title("Prediction : " + str(np.argmax(prediction)))
	plt.show()

#====================================================================================#
#This function is for preprocessing the image we get.
#====================================================================================#
def preproc(image_data) :
	res = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
	res = resize(res,(28,28), anti_aliasing=True)
	res = np.rot90(np.fliplr(res))
	res = (res)*100
	res = np.around(res)
	res[res < 20] = 0
	print(res)
	res = res/255
	return res,res[np.newaxis,:,:]
#====================================================================================#

#====================================================================================#
# The loop for detecting drawing an image and predicting what it is. #
#====================================================================================#
while True :
	#rands()
	image = get_img()
	image_data = image.get()
	res,pes = preproc(image_data)
	prediction = model.predict(pes)
	plt.grid(False)
	plt.imshow(res,cmap = plt.cm.binary)
	plt.title("Prediction : " + str(np.argmax(prediction)))
	plt.show()
#====================================================================================#


