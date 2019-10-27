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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 20 * 20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

chpt = torch.load("./classifier.pth")
model = models.vgg19(pretrained=True)
for param in model.parameters():
	param.requires_grad = False
model.classifier = Net()
model.load_state_dict(chpt['state_dict'])
for key in model:
	print(key)
#====================================================================================#


#====================================================================================#
#This function selects a random image from the test data. #
#====================================================================================#
def rands():
	rand = random.randint(0,10001)
	prediction = model.predict(test_images[rand][np.newaxis,:,:])
	plt.grid(False)
	#print(test_images[rand])
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
	res = (res)*250
	res = np.around(res)
	res[res < 20] = 0
	#print(res)
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
	prediction = model.forward(pes)
	plt.grid(False)
	plt.imshow(res,cmap = plt.cm.binary)
	print(prediction)
	plt.title("Prediction : " + str(np.argmax(prediction)))
	plt.show()
#====================================================================================#
