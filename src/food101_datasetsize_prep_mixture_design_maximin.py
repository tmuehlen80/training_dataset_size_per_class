import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


import importlib.util

spec_data_generation = importlib.util.spec_from_file_location("data_generation", "src/data_generation.py")
data_generation = importlib.util.module_from_spec(spec_data_generation)
spec_data_generation.loader.exec_module(data_generation)


# Define data transformations and load the Food101 dataset
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.Food101(root='./data', split='train', download=True, transform=transform)
train_dataset_subset = datasets.Food101(root='./data', split='train', download=True, transform=transform)
test_dataset = datasets.Food101(root='./data', split='test', download=True, transform=transform)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


n_total = train_dataset.__len__()
n_classes = len(train_dataset.classes)

#dir(train_dataset)
#train_dataset._labels

# Build a dict for each class:
data_dict = {}
masks = {}
label_dict = {}
for i in range(n_classes):
    masks[i] = pd.Series(train_dataset._labels) == i
    label_dict[i] = pd.Series(train_dataset._labels)[masks[i]]
    # data_dict[i] = train_dataset.data[masks[i],:,:,:] # this lines still needs to be updated

label_dict


n_repeat_outer = 202
subset_sizes = np.arange(5000, 10000 + 1, 5000).tolist()
#subset_sizes = np.arange(1000, 200 + 1, 1000).tolist()
n_repeat_inner = 1

n_max = subset_sizes[0]
doe = data_generation.create_constrained_mixture_design(n_classes, n_repeat_outer, n_max, 750, n_optim=5000, n_batch_size=100)

doe.shape

subsets = doe[0,:]

class_indices = {}
data_subset_dict = {}
label_subset_dict = {}

for i in range(n_classes):
    class_indices[i] = np.random.choice(750, int(subsets[i]), replace=False)
    label_subset_dict[i] = label_dict[i].iloc[class_indices[i]]
    #data_subset_dict[i] = data_dict[i][class_indices[i],:,:,:] # still missing the images ...


label_subset = pd.Series()
img_subset = []

for i in range(n_classes):
    label_subset = label_subset.append(label_subset_dict[i])
    #data_subset = np.append(data_subset, data_subset_dict[i], axis = 0)

train_dataset_subset._labels
train_dataset_subset._labels = label_subset.tolist()


#data_subset = data_subset.astype(np.uint8)
#trainset_subset.data = data_subset
#trainset_subset.targets = label_subset.tolist()
trainloader_subset = torch.utils.data.DataLoader(train_dataset_subset, batch_size=batch_size, shuffle=True, num_workers=2)

train_dataset_subset.__len__()

type(train_dataset_subset)












check_epochs = np.arange(0, 6, 2)

n_repeat = 2
accs = []
epochs_trained = []
train_losses = []
val_losses = []
itterations = []


for iteration in range(n_repeat):
    print(f'Iteration: {iteration}')
    # Use ResNet-18 as the model
    model = torchvision.models.resnet18()
    # Change the output layer to have 101 classes (for the Food101 dataset)
    model.fc = nn.Linear(model.fc.in_features, 101)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = model.to(device)
    for epoch in range(max(check_epochs) + 1):
        _ = model.train()
        running_loss = 0.0
        i = 0
        for inputs, labels in train_loader:
            i = i + 1
            if i % 50 == 0:
                print(i)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        if epoch in check_epochs:
            _ = model.eval()
            correct = 0
            total = 0
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_loss = criterion(outputs, labels)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    running_loss += val_loss.item()
            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy:.2f}%")
            itterations.append(iteration)
            epochs_trained.append(epoch)
            accs.append(100 * correct / total)
            train_losses.append(running_loss)
            val_losses.append(running_val_loss)

results = pd.DataFrame()
results["itterations"] =  itterations
results["accs"] =  accs
results["epoch"] = epochs_trained
results["train_loss"] = train_losses
results["val_loss"] = val_losses
print("writting results to file.")
results.to_csv(f"food101_full_dataset_trainings_resnet18_20230915.csv", index=False)
