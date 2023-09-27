import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from time import time
from timm import create_model

# Define data transformations and load the Food101 dataset
transform = transforms.Compose([
    #transforms.Resize((128, 128)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

check_epochs = np.arange(0, 71, 2)
#check_epochs = np.arange(0, 7, 2)

n_repeat = 10
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


#model = torchvision.models.efficientnet_v2_s()

#num_classes = 47  # Adjust based on the number of classes in EMNIST ByClass


for iteration in range(n_repeat):
    print(f'Iteration: {iteration}')
    # Use ResNet-18 as the model
    #model = EfficientNetV2Model(num_classes)
    #model.model.conv_stem = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #model = torchvision.models.resnet18()
    model = torchvision.models.mobilenet_v3_large()
    #model = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2") # is not bringing any advantage
    #model = torchvision.models.mobilenet_v3_small() # 85.07 acc after 19 epochs
    model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 47)
    #model = torchvision.models.vgg11() # works only if images are upscaled
    #model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #model.classifier[6] = nn.Linear(model.classifier[6].in_features, 47)
    #model = torchvision.models.resnet50() # works slightly better than resnet18, 85.55% acc after 9 epochs
    #model = torchvision.models.resnet152(weights="IMAGENET1K_V2")
    #model = torchvision.models.resnet152()
    #model = torchvision.models.efficientnet_v2_s()
    # Change the output layer to have 101 classes (for the Food101 dataset)
    #model.fc = nn.Linear(model.fc.in_features, 47)
    #model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45])
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = model.to(device)
    for epoch in range(max(check_epochs) + 1):
        start = time()
        _ = model.train()
        running_loss = 0.0
        i = 0
        for inputs, labels in train_loader:
            #if i %100 == 0:
            #    print(i)
            #i = i + 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #lr_scheduler.step()
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
#print("writting results to file.")
results.to_csv(f"emnistbalanced_full_dataset_trainings_mobilenet_20230921.csv", index=False)



