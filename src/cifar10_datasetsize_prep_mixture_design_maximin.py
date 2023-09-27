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

#import pyDOE

#from dexpy.samplers import uniform_simplex_sample
from scipy import spatial
#import matplotlib.pyplot as plt
import numpy as np


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
#net = models.resnet18() 
#net = models.resnet18(pretrained=True) 
#net = models.resnet34() 
#net = models.resnet34(pretrained=True) 
# net = models.vgg11_bn() # does not work, for whatever reaseon
#net = models.resnext50_32x4d(pretrained=True)
#net = models.vgg16() 

#_ = net.to(device)

#total_params = sum(	param.numel() for param in net.parameters())
#print(total_params)

# do one full run on the complete dataset to see the optimal performance:
check_epochs = np.arange(0, 200, 5)

### set hyper parameters for data generating algo:

#n_repeat = 15
n_repeat_outer = 40
subset_sizes = np.arange(5000, 45000 + 1, 5000).tolist()
#subset_sizes = np.arange(1000, 200 + 1, 1000).tolist()
n_repeat_inner = 3
# the actual big loop:
print("starting training loop for subsets:")
print(f"check epochs: {check_epochs}")
print(f"subsets: {subset_sizes}")
print(f"n_repeat_outer: {n_repeat_outer}")
print(f"n_repeat_inner: {n_repeat_inner}")
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


def uniform_simplex_sample(N, q, clip=None):
    """Returns an array of points sampled uniformly from a simplex. 
    If needed, the initial exponential rng is clipped in order to have
    a higher likelihood to fullfill a constraint.

    :param N: the number of random sample to be generated
    :param q: the dimension of the simplex
    """
    sample = np.random.exponential(1.0, (N, q))
    if clip is not None:
        sample = np.clip(sample, 0, clip)
    row_sums = sample.sum(axis=1)
    sample = sample / row_sums[:, np.newaxis]
    return sample

def create_constrained_mixture_design(d, n, n_sum, c_max, n_optim, n_batch_size):
    """Create a constrained mixture design, optimized pointwise wrt the maximin criterion."""
    n_sample = 0
    # create a first doe
    clip = None
    loop_no = 0
    while n_sample < n:
        candidate = uniform_simplex_sample(n_batch_size, d, clip=clip) 
        candidate_scaled = candidate * n_sum
        # kick out non-acceptable rows:
        matches = np.where((candidate_scaled <= c_max).sum(axis=1) == d)[0]
        if matches.shape[0] > 0:
            candidate = candidate[matches,:]
            candidate_scaled = candidate_scaled[matches,:]
            if n_sample == 0:
                if candidate.shape[0] <= n:
                    doe_best = candidate
                    doe_best_scaled = candidate_scaled
                else:
                    doe_best = candidate[:n,:]
                    doe_best_scaled = candidate_scaled[:n,:]
                n_sample = doe_best.shape[0]
            else:
                doe_best = np.append(doe_best, candidate, axis = 0)
                doe_best_scaled = np.append(doe_best_scaled, candidate_scaled, axis=0)
                n_sample = doe_best.shape[0]
        else:
            if loop_no == 0:
                clip = 10
                loop_no += 1
            clip = clip * 0.9
            print(f"clip = {clip}")

    # reduce if oversampled:
    if n_sample > n:
        doe_best = doe_best[:n, :]
        doe_best_scaled = doe_best_scaled[:n, :]
    Mm = 0
    # improve the doe iteratively:
    for i in range(n_optim):
        candidate = doe_best
        candidate_scaled = doe_best_scaled
        dm = spatial.distance_matrix(candidate, candidate)
        np.fill_diagonal(dm, 20)
        dm_argmin = dm.min(axis=1).argmin()
        n_sample = 0
        while n_sample == 0:
            candidate_rows = uniform_simplex_sample(n_batch_size, d, clip=clip) 
            candidate_rows_scaled = candidate_rows * n_sum
            # kick out non-acceptable rows:
            matches = np.where((candidate_rows_scaled <= c_max).sum(axis=1) == d)[0]
            if matches.shape[0] > 0:
                n_sample = 1
                # in this point wise exchange algo we only need one point:
                candidate_row = candidate_rows[matches[0]]
                candidate_row_scaled = candidate_rows_scaled[matches[0]]
            else:
                if loop_no == 0:
                    clip = 10
                    loop_no += 1
                clip = clip * 0.95
                print(f"clip = {clip}")
        candidate[dm_argmin,:] = candidate_row
        candidate_scaled[dm_argmin,:] = candidate_row_scaled
        if dm.min() > Mm:
            print(dm.min(), i)
            doe_best = candidate
            doe_best_scaled = candidate_scaled
            dm_best = dm
            Mm = dm.min()
    return np.round(doe_best_scaled)


for n_max in subset_sizes:
    print("optimizing doe:")
    doe = create_constrained_mixture_design(n_classes, n_repeat_outer, n_max, 5000, n_optim=500000, n_batch_size=1000)
    for k_outer in range(n_repeat_outer):
        subsets = doe[k_outer,:]
        for k_inner in range(n_repeat_inner):
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
            net.fc = nn.Linear(net.fc.in_features, 10)
            _ = net.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            #optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
            start = time()
            for epoch in range(max(check_epochs) + 1):  # loop over the dataset multiple times
                running_loss = 0.0
                _ = net.train()
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
                    _ = net.eval()                    
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
                    print(f'n_max: {n_max}, repeat_outer {k_outer}, repeat_inner {k_inner}, epoch: {epoch}, train_loss: {running_loss:.3f}, val_loss: {running_val_loss}, Accuracy: {100 * correct // total} %, current lr: {optimizer.param_groups[0]["lr"]}')
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
    results.to_csv(f"Cifar10_acc_subsets_thomas_batch_size_512_reset_fc_output_size_mixture_design_maximin_subsetsize_{n_max}_20230921.csv", index=False)



print("writting results to file.")
results = pd.DataFrame()
results["accs"] = accs
results["training_times"] = times
for i, c in enumerate(classes):
    results[c] = subsets_collected[:, i]
results["epochs_trained"] = epochs_trained
results.to_csv("Cifar10_acc_subsets_thomas_batch_size_512_reset_fc_output_size_mixture_design_maximin_20230921.csv", index=False)
