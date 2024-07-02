import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import models.cifar

device = torch.device("cuda:0")

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=0)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=0)

writer = SummaryWriter(log_dir='tensorLog')


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # x = F.relu(F.max_pool2d(F.dropout(self.conv1(x), training=True), 2))
        # x = F.relu(F.max_pool2d(F.dropout(self.conv2(x), training=True), 2))
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.05, training=True)
        x = self.fc2(x)
        return x


def mcdropout_test(model):
    model.train()
    test_loss = 0
    correct = 0
    T = 50
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output_list = []
            for i in range(T):
                output_list.append(torch.unsqueeze(model(data), 0))
            output_mean = torch.cat(output_list, 0).mean(0)
            test_loss += criterion(output_mean, target)  # sum up batch loss
            pred = output_mean.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nMC Dropout Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            model.eval()
            data, target = data.cuda(), target.cuda()
            output_list = []
            for i in range(1):
                output_list.append(torch.unsqueeze(model(data), 0))
            output_mean = torch.cat(output_list, 0).mean(0)
            output = output_mean
            test_loss += criterion(output, target)  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

model_path = "checkpoint/mnist/cnn.pth"

# 加载模型参数
if os.path.exists(model_path):
    model = CNN_Model()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    print("Model loaded successfully.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    test(model)
    mcdropout_test(model)
else:
    model = CNN_Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 300 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'
                      .format(epoch, batch_idx * len(data),
                              len(train_loader.dataset),
                              100. * batch_idx / len(train_loader),
                              loss.item()))
        writer.add_scalar('Train/Loss', 2, epoch)

    checkpoint = {
        'epoch': 3,
        'state_dict': model.state_dict()
    }
    filepath = 'checkpoint/mnist'

    torch.save(model.state_dict(), os.path.join(filepath, "cnn.pth"))

    test(model)
    mcdropout_test(model)



