import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

from time import time

import torch

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models

import pyDOE

from dexpy.samplers import uniform_simplex_sample
from scipy import spatial
import matplotlib.pyplot as plt
import numpy as np


# for Thomas: please use venv_events_3810 for running this script

# Prepare cifar dataset:
print("prepare datasets")


# please take care that the same transforms are used in full training and in the algo below.

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, 4),   
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]     
     )

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]     
     )


batch_size = 512
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# load data again as shell:
trainset_subset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_total = trainset_subset.data.shape[0]
n_classes = pd.Series(trainset_subset.targets).nunique()

# Build a dict for each class:
# needed here?
data_dict = {}
masks = {}
label_dict = {}
for i in range(n_classes):
    masks[i] = pd.Series(trainset_subset.targets) == i
    label_dict[i] = pd.Series(trainset_subset.targets)[masks[i]]
    data_dict[i] = trainset_subset.data[masks[i],:,:,:]



# now do the actual training:
print("doing full training.")

# the choice of the model is somehow subjective,
# keeping it as small as possible while still showing some performance.
# Resnet18 seems to fullfill this.

# do one full run on the complete dataset to see the optimal performance:
check_epochs = np.arange(0, 201, 5)

criterion = nn.CrossEntropyLoss()
# number of times the full training should be repeated.
n_repeat = 10
# create lists for storing different kinds of results
accs = []
epochs_trained = []
train_losses = []
val_losses = []
itterations = []

start = time()
for iteration in range(n_repeat):
    print(f'Iteration: {iteration}')
    net = models.resnet18() 
    net.fc = nn.Linear(net.fc.in_features, 10)
    _ = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
    total_params = sum(	param.numel() for param in net.parameters())
    # doing a standard pytorch training approach:
    for epoch in range(max(check_epochs) + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        _ = net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        lr_scheduler.step()
        results = pd.DataFrame()
        # calculating performance on test dataset:
        if epoch in check_epochs:
            _ = net.eval()
            # calculate test acc:
            correct = 0
            total = 0
            running_val_loss = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    val_loss = criterion(outputs, labels)
                    running_val_loss += val_loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f' epoch: {epoch}, train_loss: {running_loss:.3f}, val_loss: {running_val_loss}, Accuracy: {100 * correct // total} %, current lr: {optimizer.param_groups[0]["lr"]}')
            itterations.append(iteration)
            epochs_trained.append(epoch)
            accs.append(100 * correct / total)
            train_losses.append(running_loss)
            val_losses.append(running_val_loss)

results["itterations"] = itterations
results["accs"] =  accs
results["epoch"] = epochs_trained
results["train_loss"] = train_losses
results["val_loss"] = val_losses
print("writting results to file.")
results.to_csv(f"Cifar10_acc_full_train_resnet_fc_reset_to_outputsize_20_iterations_20230921.csv", index=False)
end = time()
print(end - start)

