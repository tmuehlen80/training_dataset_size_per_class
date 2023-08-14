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

import pyDOE


# for Thomas: please use venv_events_3810 for running this script

# Prepare cifar dataset:
print("prepare datasets")


# please take care that the same transforms are used in full training and in the algo below.

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, 4),   
     #transforms.ColorJitter(),
     #transforms.RandomPerspective(),  
     #transforms.Resize(size=(224, 224)), # necessary only for VGG16
     transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]     
     )

transform_test = transforms.Compose(
    [#transforms.Resize(size=(224, 224)), # necessary only for VGG16
     transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]     
     )



#batch_size = 256
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
#print(n_total, n_classes)

# Build a dict for each class:
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

#net = models.mobilenet_v3_small() # best
net = models.resnet18() 
#net = models.resnet18(pretrained=True) 
#net = models.resnet34() 
#net = models.resnet34(pretrained=True) 
# net = models.vgg11_bn() # does not work, for whatever reaseon
#net = models.resnext50_32x4d(pretrained=True)
#net = models.vgg16() 

_ = net.to(device)

total_params = sum(	param.numel() for param in net.parameters())
print(total_params)

# do one full run on the complete dataset to see the optimal performance:
check_epochs = np.arange(0, 200, 5)
#check_epochs = [3, 6]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
#optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=1e-4)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

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
        print(f' epoch: {epoch}, train_loss: {running_loss:.3f}, val_loss: {running_val_loss}, Accuracy: {100 * correct // total} %, current lr: {optimizer.param_groups[0]["lr"]}')

end = time()
print(end - start)


### set hyper parameters for data generating algo:

n_repeat = 20
subset_sizes = np.arange(1000, 4500 + 1, 500).tolist()


# the actual big loop:
print("starting training loop for subsets:")
print(f"check epochs: {check_epochs}")
print(f"subsets: {subset_sizes}")
print(f"n_repeat: {n_repeat}")
subsets_collected = np.empty((0, n_classes))
accs = []
epochs_trained = []
times = []

def sample_cond_unif(n_classes, target_sum):
    """Sample randomly a subset with a targeted overall sum of images."""
    proportions = np.random.sample(n_classes)
    subsets = np.round(target_sum * proportions / proportions.sum())
    return subsets


def determine_sample_distr(n_classes, n_max, class_max_counts):
    """Correct a given subset for the maximum number of images in a class."""
    subsets = sample_cond_unif(n_classes, n_max)
    while (subsets > class_max_counts).sum() > 0:
        argmax = np.argmax(subsets)
        overhead = subsets[argmax] - class_max_counts
        arg_smaller_class_max_counts = np.where(subsets < class_max_counts)
        tmp = sample_cond_unif(arg_smaller_class_max_counts[0].shape, overhead)
        subsets[arg_smaller_class_max_counts[0]] = subsets[arg_smaller_class_max_counts[0]] + tmp
        subsets[argmax] = subsets[argmax] - tmp.sum()
        #print(subsets)
    return subsets



for n_max in subset_sizes:
    #print(n_max)
    # lhs = pyDOE.lhs(n_classes, n_repeat, criterion="maximin", iterations=2000)
    for k in range(n_repeat):
        #print(k)
        proportions = np.random.sample(n_classes)
        subsets = determine_sample_distr(n_classes, n_max, 5000)
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
        net = models.resnet18() 
        _ = net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        #optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
        start = time()
        for epoch in range(max(check_epochs) + 1):  # loop over the dataset multiple times
            running_loss = 0.0
            #for i, data in enumerate(trainloader, 0):
            for i, data in enumerate(trainloader_subset, 0):
                # get the inputs; data is a list of [inputs, labels]
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
            if epoch in check_epochs:
                end = time()
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
                print(f'n_max: {n_max}, repeat {k}, epoch: {epoch}, train_loss: {running_loss:.3f}, val_loss: {running_val_loss}, Accuracy: {100 * correct // total} %, current lr: {optimizer.param_groups[0]["lr"]}')
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
results.to_csv("Cifar10_acc_subsets_thomas_batch_size_512_mixture_design_20230704.csv", index=False)