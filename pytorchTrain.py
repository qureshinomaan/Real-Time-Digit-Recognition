import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#=========================================================#
# Loading the dataset.
# transforms.ToTensor converts the pil image to image tensor.
# Channel in pytorch
	# Example RGB has 3 channels.
	# Greyscale has 1 channel.
# transforms.Normalise((m1,m2,m3), (d1, d2, d3))
# m1, m2, m3 is mean for each channel.
# d1, d2, d3 is standard deviation. for each channel.
#=========================================================#
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

trainset = torchvision.datasets.MNIST(
      root='./data/mnist', 
			train=True,
			download=True, 
			transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, 
      batch_size=4,
      shuffle=True, 
      num_workers=2)

testset = torchvision.datasets.MNIST(
      root='./data/mnist', 
      train=False,
      download=True, 
      transform=transform)

testloader = torch.utils.data.DataLoader(testset, 
      batch_size=4,
      shuffle=False, 
      num_workers=2)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

#=========================================================#
examples = enumerate(trainloader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_targets.shape)

# def imshow(img):
# 	img = img / 2 + 0.5  #unnormalise
# 	npimg = img.numpy()
# 	plt.imshow(np.transpose(npimg, (1, 2, 0))) 
# 	plt.show()


# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# # imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#=========================================================#

#=========================================================#
# Defining the neural network.
#=========================================================#
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
        x = F.relu(self.fc3(x))
        return x


net = Net()


#=========================================================#

#=========================================================#
# Defining the Neural Network Loss Function. 
#=========================================================#
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)
#=========================================================#

#=========================================================#
# Training the model. 
#=========================================================#
# for epoch in range(1):
#   running_loss = 0.0
#   for i, data in enumerate(trainloader):
#     inputs, label = data
#     optimizer.zero_grad()
#     outputs = net(inputs)
#     loss = criterion(outputs, label)
#     loss.backward()
#     optimizer.step()
#     running_loss += loss.item()
#     if i % 2000 == 1999:    # print every 2000 mini-batches
#         print('[%d, %5d] loss: %.3f' %
#         (epoch + 1, i + 1, running_loss / 2000))
#         running_loss = 0.0
#=========================================================#

for i, data in enumerate(testloader):
  inputs, label = data 
  outputs = net(inputs)
  print("outputs : ", outputs)
  print("Lables : ", label)
  value, indices = torch.max(outputs[0], 0)
  print(indices)
  input()

#=========================================================#
# Saving the Pytorch Model.
#=========================================================#
torch.save({'arch' : 'vgg16', 'state_dict' : net.state_dict},'classifier.pth')
#=========================================================#


