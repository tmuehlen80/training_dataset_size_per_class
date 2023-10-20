import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from time import time

# Define data transformations and load the Food101 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.Food101(root='./data', split='train', download=True, transform=transform)
test_dataset = datasets.Food101(root='./data', split='test', download=True, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

check_epochs = np.arange(0, 101, 5)
#check_epochs = np.arange(0, 7, 2)

n_repeat = 2
accs = []
epochs_trained = []
train_losses = []
val_losses = []
itterations = []

#model = torchvision.models.resnet152(weights="IMAGENET1K_V2")
#model.fc = nn.Linear(model.fc.in_features, 101)

#total_params = sum(	param.numel() for param in model.parameters())
#print(total_params)


#sum(param.numel() for param in model.layer4.parameters())

#len(list(model.layer1.parameters()))
#model.train()

# training time without freezing layers, batchsize 128, pretrained:
#Epoch 1, Loss: 2.002537102111288
#time per epoch (secs): 384.1391680240631
#Test Accuracy: 60.92%


# training time with freezing layers, batchsize 128, pretrained:
#Epoch 1, Loss: 2.0664357289269164
#time per epoch (secs): 260.2132399082184
#Test Accuracy: 62.80%

# for param in model.conv1.parameters():
#     param.requires_grad = False

# for param in model.bn1.parameters():
#     param.requires_grad = False

# for param in model.layer1.parameters():
#     param.requires_grad = False

# for param in model.layer2.parameters():
#     param.requires_grad = False

# for param in model.layer3.parameters():
#     param.requires_grad = False

#n_free_param = 0
#for param in model.parameters():
#    if param.requires_grad:
#        n_free_param = n_free_param + param.numel()

#print(n_free_param)

for iteration in range(n_repeat):
    print(f'Iteration: {iteration}')
    # Use ResNet-18 as the model
    #model = torchvision.models.resnet18()
    #model = torchvision.models.resnet152(weights="IMAGENET1K_V2")
    model = torchvision.models.resnet152()
    #model = torchvision.models.efficientnet_v2_s()
    # Change the output layer to have 101 classes (for the Food101 dataset)
    model.fc = nn.Linear(model.fc.in_features, 101)
    # for param in model.conv1.parameters():
    #     param.requires_grad = False
    # for param in model.bn1.parameters():
    #     param.requires_grad = False
    # for param in model.layer1.parameters():
    #     param.requires_grad = False
    # for param in model.layer2.parameters():
    #     param.requires_grad = False
    # for param in model.layer3.parameters():
    #     param.requires_grad = False
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = model.to(device)
    for epoch in range(max(check_epochs) + 1):
        start = time()
        _ = model.train()
        running_loss = 0.0
        #i = 0
        for inputs, labels in train_loader:
            #if i %50 == 0:
            #    print(i)
            #i = i + 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        stop = time()
        print(f"time per epoch (secs): {stop - start}")
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
results["itterations"] = itterations
results["accs"] =  accs
results["epoch"] = epochs_trained
results["train_loss"] = train_losses
results["val_loss"] = val_losses
print("writting results to file.")
results.to_csv(f"food101_full_dataset_trainings_resnet152_20230915.csv", index=False)
