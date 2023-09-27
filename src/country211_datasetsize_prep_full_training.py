import pandas as pd
#import warnings
#from pandas.core.common import SettingWithCopyWarning
#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
#warnings.simplefilter(action="ignore", category=FutureWarning)

from time import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
#import pyDOE
#from dexpy.samplers import uniform_simplex_sample
#from scipy import spatial
#import matplotlib.pyplot as plt
import numpy as np

# https://www.kaggle.com/code/mdreyadhossainnsu/food-101-classify-with-pytorch
# https://github.com/shubhajitml/food-101/blob/master/food-101-pytorch.ipynb

# for Thomas: please use venv_events_3810 for running this script

# Prepare cifar dataset:
print("prepare datasets")


batch_size = 64
#batch_size = 2048
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# load data again as shell:
# Write transform for image
# transform from https://www.kaggle.com/code/mdreyadhossainnsu/food-101-classify-with-pytorch
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(128, 128)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# imagenet averages
])

data_transform_test = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(128, 128)),
    # Turn the image into a torch.Tensor
    transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.Country211(root='./data', split="train", download=True, transform=data_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.Country211(root='./data', split="test", download=True, transform=data_transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)


check_epochs = np.arange(0, 200, 5)

criterion = nn.CrossEntropyLoss()


accs = []
epochs_trained = []
train_losses = []
val_losses = []
itterations = []

#net = models.resnet18() 
net = models.resnet152() 
_ = net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
total_params = sum(	param.numel() for param in net.parameters())
# print(total_params)


for epoch in range(max(check_epochs) + 1):  # loop over the dataset multiple times
    running_loss = 0.0
    #for i, data in enumerate(trainloader, 0):
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        # in case of using a GPU:
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
    lr_scheduler.step()
    #if i % 100 == 99:    # print every 2000 mini-batches
    #results = pd.DataFrame()
    if epoch in check_epochs:
        # calculate test acc:
        correct = 0
        total = 0
        running_val_loss = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f' epoch: {epoch}, train_loss: {running_loss:.3f}, val_loss: {running_val_loss}, Accuracy: {100 * correct // total} %, current lr: {optimizer.param_groups[0]["lr"]}')
        epochs_trained.append(epoch)
        accs.append(100 * correct / total)
        train_losses.append(running_loss)
        val_losses.append(running_val_loss)

