import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from time import time
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


import importlib.util

spec_data_generation = importlib.util.spec_from_file_location("data_generation", "src/data_generation.py")
data_generation = importlib.util.module_from_spec(spec_data_generation)
spec_data_generation.loader.exec_module(data_generation)



# Define data transformations and load the Food101 dataset
transform = transforms.Compose([
    #transforms.Resize((128, 128)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
train_dataset_subset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)


batch_size = 512
check_epochs = np.arange(0, 91, 2)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


n_total = train_dataset.__len__()
n_classes = len(train_dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Build a dict for each class:
# this is needed later for randomly sampling from each class.
data_dict = {}
masks = {}
label_dict = {}
for i in range(n_classes):
    masks[i] = pd.Series(train_dataset.targets) == i
    label_dict[i] = pd.Series(train_dataset.targets)[masks[i]]
    data_dict[i] = train_dataset.data[masks[i],:,:]



n_repeat_outer = 140
subset_sizes = np.arange(10000, 100000 + 1, 10000).tolist()
# if desired, for each row of the doe created, the training can be repeated:
n_repeat_inner = 1


subsets_collected = np.empty((0, n_classes))
accs = []
epochs_trained = []
times = []

for n_max in subset_sizes:
    #n_max = subset_sizes[0]
    print("optimizing doe:")
    doe = data_generation.create_constrained_mixture_design(n_classes, n_repeat_outer, n_max, 2400, n_optim=100000, n_batch_size=1000)
    for k_outer in range(n_repeat_outer):
        #k_outer = 0
        subsets = doe[k_outer,:]
        for k_inner in range(n_repeat_inner):
            #k_inner = 0
            class_indices = {}
            data_subset_dict = {}
            label_subset_dict = {}
            for i in range(n_classes):
                class_indices[i] = np.random.choice(2400, int(subsets[i]), replace=False)
                label_subset_dict[i] = label_dict[i].iloc[class_indices[i]]
                data_subset_dict[i] = data_dict[i][class_indices[i],:,:]
            # now put the subset dataset together again and shuffle:
            label_subset = pd.Series()
            data_subset = np.empty((0, 28, 28))
            for i in range(n_classes):
                label_subset = label_subset.append(label_subset_dict[i])
                data_subset = np.append(data_subset, data_subset_dict[i], axis = 0)
            data_subset = data_subset.astype(np.uint8) # needed? This is the datatype in the unchanged data object, however, it is somehow strange, considering the transform to stay with int, and not change to float.
            #train_dataset_subset.data = data_subset
            train_dataset_subset.data = torch.from_numpy(data_subset)
            #train_dataset_subset.targets = label_subset.tolist()
            train_dataset_subset.targets = torch.from_numpy(label_subset.to_numpy())
            trainloader_subset = DataLoader(train_dataset_subset, batch_size=batch_size, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            # now do the actual training:
            net = torchvision.models.mobilenet_v3_large()
            net.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            net.classifier[3] = nn.Linear(net.classifier[3].in_features, 47)
            _ = net.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            #optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=1e-4)
            #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
            start = time()
            for epoch in range(max(check_epochs) + 1):  # loop over the dataset multiple times
                running_loss = 0.0
                _ = net.train()
                #batch = trainloader_subset
                #for i, data in enumerate(trainloader, 0):
                for inputs, labels in trainloader_subset:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # get the inputs; data is a list of [inputs, labels]
                    #inputs, labels = data[0].to(device), data[1].to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    # print statistics
                    running_loss += loss.item()
                #lr_scheduler.step()
                if epoch in check_epochs:
                    _ = net.eval()                    
                    end = time()
                    subsets_collected = np.append(subsets_collected, subsets.reshape(1, -1), axis=0)
                    epochs_trained.append(epoch)
                    # calculate test acc:
                    correct = 0
                    total = 0
                    running_val_loss = 0
                    # since we'rnot training, we don't need to calculate the gradients for our outputs
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            #images, labels = data
                            # images, labels = data[0].to(device), data[1].to(device)
                            # calculate outputs by running images through the network
                            outputs = net(inputs)
                            # the class with the highest energy is what we choose as prediction
                            _, predicted = torch.max(outputs, 1)
                            val_loss = criterion(outputs, labels)
                            running_val_loss += val_loss.item()
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    print(f'n_max: {n_max}, repeat_outer {k_outer}, repeat_inner {k_inner}, epoch: {epoch}, train_loss: {running_loss:.3f}, val_loss: {running_val_loss}, Accuracy: {100 * correct // total} %, current lr: {optimizer.param_groups[0]["lr"]}')
                    times.append(end-start)
                    start = time()
                    accs.append(correct / total)


# end loop
print("writting results to file.")
results = pd.DataFrame()
results["accs"] = accs
results["training_times"] = times
for i, c in enumerate(train_dataset.classes):
    results[c] = subsets_collected[:, i]

results["epochs_trained"] = epochs_trained
results.to_csv(f"emnist_acc_subsets_thomas_batch_size_512_mobilenet_mixture_design_maximin_subsetsize_{n_max}_20230922_.csv", index=False)



