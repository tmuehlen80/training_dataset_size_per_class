import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

from time import time

#from tqdm import tqdm
import torch

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models

# for Thomas: please use venv_events_3810 for running this script

# Prepare cifar dataset:
print("prepare datasets")
#transform = transforms.Compose(
#    [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#batch_size = 512 # not actually used, just for loading images

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#print(dir(trainset))
#print(len(trainset.targets))
#trainset.data.shape
#n_total = trainset.data.shape[0]
#n_classes = pd.Series(trainset.targets).nunique()
#print(n_total, n_classes)

# Build a dict for each class:
#data_dict = {}
#masks = {}
#label_dict = {}
#for i in range(n_classes):
#    masks[i] = pd.Series(trainset.targets) == i
#    label_dict[i] = pd.Series(trainset.targets)[masks[i]]
#    data_dict[i] = trainset.data[masks[i],:,:,:]





transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, 4),   
     #transforms.RandomGrayscale(0.4), 
     #transforms.ColorJitter(),
     #transforms.RandomPerspective(),  
     transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]     
     )

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]     
     )



#transform = transforms.Compose(
#    [transforms.ToTensor(),
#     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])]     
#     )


#batch_size = 256
batch_size = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# load data again as shell:
trainset_subset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# now do the actual training:

#net = models.mobilenet_v3_small() # best
net = models.resnet18() 

_ = net.to(device)

total_params = sum(	param.numel() for param in net.parameters())
print(total_params)

# do one full run on the complete dataset to see the optimal performance:
print("doing full training.")
check_epochs = np.arange(0, 101, 5)
#check_epochs = [3, 6]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75])

start = time()
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
        print(f' epoch: {epoch + 1}, train_loss: {running_loss:.3f}, val_loss: {running_val_loss}, Accuracy: {100 * correct // total} %, current lr: {optimizer.param_groups[0]["lr"]}')

end = time()
print(end - start)



### set hyper parameters

n_repeat = 20
subset_sizes = np.arange(1000, 4000 + 1, 500).tolist()
subset_sizes

# the actual big loop:
print("starting training loop for subsets:")
print(f"check epochs: {check_epochs}")
print(f"subsets: {subset_sizes}")
print(f"n_repeat: {n_repeat}")
subsets_collected = np.empty((0, n_classes))
accs = []
epochs_trained = []
times = []
max(check_epochs)


for n_max in subset_sizes:
    #print(n_max)
    for k in range(n_repeat):
        #print(k)
        proportions = np.random.sample(n_classes)
        #proportions = proportions / proportions.sum()
        subsets = np.round(proportions*n_max)
        #subsets_collected.append(subsets)
        #subsets_collected = np.append(subsets_collected, subsets.reshape(1, -1), axis=0)
        class_indices = {}
        data_subset_dict = {}
        label_subset_dict = {}
        for i in range(n_classes):
            class_indices[i] = np.random.choice(5000, int(subsets[i]), replace=False)
            label_subset_dict[i] = label_dict[i].iloc[class_indices[i]]
            data_subset_dict[i] = data_dict[i][class_indices[i],:,:,:]
        # now put the subset dataset together again and shuffle:
        label_subset = pd.Series()
        data_subset = np.empty((0, 32, 32, 3))
        for i in range(n_classes):
            label_subset = label_subset.append(label_subset_dict[i])
            data_subset = np.append(data_subset, data_subset_dict[i], axis = 0)
        data_subset = data_subset.astype(np.uint8)
        trainset_subset.data = data_subset
        trainset_subset.targets = label_subset.tolist()
        trainloader_subset = torch.utils.data.DataLoader(trainset_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        # now do the actual training:
        # net = Net()
        net = models.mobilenet_v3_small() 
        _ = net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
        start = time()
        for epoch in range(max(check_epochs) + 1):  # loop over the dataset multiple times
            running_loss = 0.0
            #for i, data in enumerate(trainloader, 0):
            for i, data in enumerate(trainloader_subset, 0):
                # get the inputs; data is a list of [inputs, labels]
                #inputs, labels = data
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
                #if i % 100 == 99:    # print every 2000 mini-batches
            if epoch in check_epochs:
                end = time()
                #print(f'[{epoch + 1}] loss: {running_loss:.3f}')
                subsets_collected = np.append(subsets_collected, subsets.reshape(1, -1), axis=0)
                epochs_trained.append(epoch)
                # calculate test acc:
                correct = 0
                total = 0
                # since we'rnot training, we don't need to calculate the gradients for our outputs
                with torch.no_grad():
                    for data in testloader:
                        #images, labels = data
                        images, labels = data[0].to(device), data[1].to(device)
                        # calculate outputs by running images through the network
                        outputs = net(images)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                print(f'n_max: {n_max}, repeat {k}, epoch: {epoch + 1}, loss: {running_loss:.3f}, Accuracy: {100 * correct // total} %')
                times.append(end-start)
                start = time()
                accs.append(correct / total)


print("writting results to file.")
results = pd.DataFrame()
results["accs"] = accs
results["training_times"] = times
for i, c in enumerate(classes):
    results[c] = subsets_collected[:, i]
results["epochs_trained"] = epochs_trained
results.to_csv("Cifar10_acc_subsets_20230615.csv", index=False)







##########################################
### junkyard ###
##########################################

# one possible todo is to find a model architecture which reaches higher accuracies.
# This medium post could be an option for a better model
#class Net(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        #self.conv3 = nn.Conv2d(16, 32, 5)
#        #self.conv4 = nn.Conv2d(32, 16, 5)
#        self.fc1 = nn.Linear(16 * 5 * 5, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 10)
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        #x = self.pool(F.relu(self.conv3(x)))
#        #x = self.pool(F.relu(self.conv4(x)))
#        x = torch.flatten(x, 1) # flatten all dimensions except batch
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x


#net = Net()
#net = models.densenet121()
#net = models.resnet50()
#net = models.resnet18()
#net = models.densenet201()
#net = models.inception_v3()
##net = models.mobilenet_v2()
#net = models.efficientnet_v2_s()
#net = models.vgg11()

