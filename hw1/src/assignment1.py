# -*- coding: utf-8 -*-
"""assignment1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QA1MKu8zopOWqPwYL9yacC2xxVLtruKn
"""

# Install dependencies
# !pip install seaborn
# !pip install torch
# !pip install torchvision

# Import modules
import seaborn as sns
import numpy as np
import math
import torch
import torchvision
import torch.utils.data as data
from torch.utils.data import SubsetRandomSampler

# Download data
CIFAR_MEAN = 0
CIFAR_STDEV = 1

def get_cifar(get_train_set):
    return torchvision.datasets.CIFAR10(
        root="data",
        train=get_train_set,
        download=True,
        # For demonstration
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((CIFAR_MEAN,), (CIFAR_STDEV,)),
            ]
        ),
    )


def main():
    trn_set = get_cifar(get_train_set=True)
    valtr_set = get_cifar(get_train_set=False)

    # Split all_test_set into 50% validation and 50% test
    # "dataset" is CIFAR object returned by get_cifar

    val_size = 0.5  # 50% of the  test set
    num_samples = len(valtr_set)
    split_idx = int(num_samples * val_size)


all_indices = np.random.choice(num_samples, num_samples, replace=False)
val_indices = all_indices[:split_idx]
test_indices = all_indices[split_idx:]


val_set = data.Subset(valtr_set, val_indices)
test_set = data.Subset(valtr_set, test_indices)

# Check data split
print(
    f"We have {len(trn_set)} train images, {len(val_set)} validation images, {len(test_set)} test images"
)

# Data loaders
BATCH_SIZE = 32
trn_loader = data.DataLoader(trn_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Let's initialize our model
N_CLASSES = 10
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Could use Sequential
        self.layer_1 = torch.nn.Linear(32 * 32 * 3, 120)
        self.layer_2 = torch.nn.Linear(120, 64)
        self.layer_3 = torch.nn.Linear(64, N_CLASSES)

    def forward(self, x):
        # print('before reshape: ',x.shape)
        x = x.view(-1, 3 * 32 * 32)
        # print('after reshape: ', x.shape)
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        return self.layer_3(x)


# Model
model = MLP()

# Let's use GPU! 16 GB NVIDIA Tesla T4
model = model.cuda()

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Time to train
NUM_EPOCHS = 10

trn_accs = []
val_accs = []
trn_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    running_correct = 0
    for batch in trn_loader:
        inputs, labels = batch
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(torch.argmax(outputs, dim=-1) == labels).item()

    # Validate
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            val_running_correct += torch.sum(
                torch.argmax(outputs, dim=-1) == labels
            ).item()

    print(f"Epoch: {epoch+1}/{NUM_EPOCHS}", end=" ")

    avg_trn_loss = running_loss / len(trn_loader)
    avg_val_loss = val_running_loss / len(val_loader)
    trn_losses.append(avg_trn_loss)
    val_losses.append(avg_val_loss)

    trn_acc = running_correct / len(trn_set)
    val_acc = val_running_correct / len(val_set)
    trn_accs.append(trn_acc)
    val_accs.append(val_acc)

    print(
        f"Loss: {round(avg_trn_loss, 4)} ({round(avg_val_loss, 4)})", end=" "
    )  # Avg. per batch
    print(f"Accuracy: {round(trn_acc, 2)} ({round(val_acc, 2)})")

print("Finished Training")

# Let's check out losses and accuracies
import matplotlib.pyplot as plt

plt.plot(range(NUM_EPOCHS), trn_accs)
plt.plot(range(NUM_EPOCHS), val_accs)
plt.show()

# Set model to evaluation mode ()
# evaluate accuracy https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        images, labels = batch

        # assign all inputs/labels to gpu
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Accuracy of the network on the 5000 test images: {100 * correct // total} %")

"""# 2. Convolutional Neural Network



W = width, F = filter size, P is padding, S is stride
"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 6, 5
        )  # (3 is input channels) (6 is output channel) (5 is kernel size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            6, 16, 5
        )  # (6 is input channel) (16 is output channel) (5 is kernel size)
        self.conv3 = nn.Conv2d(16, 32, 2)
        self.fc1 = nn.Linear(32 * 2 * 2, 120)  # need to check:
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten the output of the last conv+pool
        x = F.relu(self.fc1(x))  # pass x through 1st fully connected layer
        x = F.relu(self.fc2(x))  # pass x through 2nd,
        x = self.fc3(x)
        return x


net = Net().cuda()

# Initialize CNN. Make sure to call optimizer with net.parameters()!

criterion = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

"""## 2a. train cnn with 3 layers"""

NUM_EPOCHS = 10

trn_accs = []
val_accs = []
trn_losses = []
val_losses = []
running_loss = 0.0

for epoch in range(NUM_EPOCHS):
    for batch in trn_loader:
        # get the image and labels from dataloader
        image, labels = batch
        image = image.cuda()
        labels = labels.cuda()

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = net(image)
        loss = criterion(outputs, labels)

        # backwards propogation, optimization step
        loss.backward()
        optimizer.step()  # how does optimizer know what the MLE is?

        # statistics
        running_loss += loss.item()
    print(f"[{epoch + 1}], loss: {running_loss / BATCH_SIZE:.3f}")
    running_loss = 0.0
print("Training Completed")

"""## 2b. tune cnn with 3 layers with validation"""

net.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        images, labels = batch

        # assign all inputs/labels to gpu
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Accuracy of the network on the 5000 test images: {100 * correct // total} %")

"""# 3. Resnet 18 implementation"""

import torchvision.models as models

# use pre-defined model and forward prop
resnet18 = models.resnet18()
resnet18.fc = nn.Linear(resnet18.fc.in_features, N_CLASSES)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(resnet18.parameters(), lr=LEARNING_RATE)

resenet18 = resnet18.cuda()

for epoch in range(NUM_EPOCHS):
    for batch in trn_loader:
        # get the image and labels from dataloader
        image, labels = batch
        image = image.cuda()
        labels = labels.cuda()

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = resnet18(image)
        loss = criterion(outputs, labels)

        # backwards propogation, optimization step
        loss.backward()
        optimizer.step()  # how does optimizer know what the MLE is?

        # statistics
        running_loss += loss.item()
    print(f"[{epoch + 1}], loss: {running_loss / BATCH_SIZE:.3f}")
    running_loss = 0.0
print("Training Completed")

resnet18.eval()
correct = 0
total = 0

# test accuracy on validation
with torch.no_grad():
    for batch in val_loader:
        images, labels = batch

        # assign all inputs/labels to gpu
        images = images.cuda()
        labels = labels.cuda()
        outputs = resnet18(images)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Accuracy of the network on the 5000 test images: {100 * correct // total} %")

"""# 4. Experiments with Resnet18 and data augmentation"""

torch.manual_seed(17)
from torchvision import transforms

# Define a sequence of transforms (as recommended in Pytorch docs)
transforms = torch.nn.Sequential(
    transforms.CenterCrop(10),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
scripted_transforms = torch.jit(transforms)

# create augmented train_dataset
trn_aug_set = torchvision.datasets.CIFAR10(
    root="data", train=True, transform=scripted_transforms, download=True
)

# create dataloader for trn__aug_dataset
trn_aug_loader = data.DataLoader(trn_aug_set, batch_size=BATCH_SIZE, shuffle=True)

# initialize a new resnet18 instance

# use pre-defined model and forward prop
resnet18_aug = models.resnet18()
resnet18_aug.fc = nn.Linear(resnet18_aug.fc.in_features, N_CLASSES)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(resnet18_aug.parameters(), lr=LEARNING_RATE)

# Train on augmented images
for epoch in range(NUM_EPOCHS):
    for batch in trn_aug_loader:
        # get the image and labels from dataloader
        image, labels = batch
        image = image.cuda()
        labels = labels.cuda()

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = resnet18_aug(image)
        loss = criterion(outputs, labels)

        # backwards propogation, optimization step
        loss.backward()
        optimizer.step()  # how does optimizer know what the MLE is?

        # statistics
        running_loss += loss.item()
    print(f"[{epoch + 1}], loss: {running_loss / BATCH_SIZE:.3f}")
    running_loss = 0.0
print("Training Completed")

# test accuracy on validation
with torch.no_grad():
    for batch in val_loader:
        images, labels = batch

        # assign all inputs/labels to gpu
        images = images.cuda()
        labels = labels.cuda()
        outputs = resnet18_aug(images)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Accuracy of the network on the 5000 test images: {100 * correct // total} %")
