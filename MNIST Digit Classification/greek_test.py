# 
# Avnish Patel
#



# import statements
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import utility
import mnist
import numpy as np
torch.backends.cudnn.enabled = False

# A deep network with the following layers:
# A convolution layer with 10 5x5 filters
# A max pooling layer with a 2x2 window and a ReLU function applied.
# A convolution layer with 20 5x5 filters
# A dropout layer with a 0.5 dropout rate (50%)
# A max pooling layer with a 2x2 window and a ReLU function applied
# A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
# A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
class MyNetwork(nn.Module):
    # initialize the network layers
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # compute a forward pass for the network
    #defines the computation that will be performed when you pass input data (x) through the model.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # relu on max pooled results of conv1
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # relu on max pooled results of dropout of conv2
        x = x.view(-1, 320)  # flatten operation
        x = F.relu(self.fc1(x))  # relu on fully connected linear layer with 50 nodes
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)  # fully connect linear layer with 10 nodes
        return F.log_softmax(x, 1)  # apply log_softmax()


# Load pre-trained model


network = torch.load('E:/Computer_Vision/Project6/model.pth')
network.eval()
for param in network.parameters():
    param.requires_grad = False

import os

image_dir = "E:/Computer_Vision/Project6/greek_test"
    # It applies some image transformations using transforms.Compose, including resizing the images to (28, 28), converting them to grayscale,
    #  inverting them, converting them to PyTorch tensors, and normalizing them.
greek_test_loader = datasets.ImageFolder(image_dir,
                                 transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                transforms.Grayscale(),
                                                                 transforms.functional.invert,
                                                                transforms.ToTensor(),
                                                               transforms.Normalize((0.1307,), (0.3081,))]))


from PIL import Image
def plot_predictions(data_dir, total, row, col):
    fig, axs = plt.subplots(row, col)
    labels = {'alpha': 0, 'beta': 1, 'gamma': 2}
    for i, folder in enumerate(['alpha', 'beta', 'gamma']):
        img_paths = os.listdir(os.path.join(data_dir, folder))
        for j, img_path in enumerate(img_paths[:total]):
            img = Image.open(os.path.join(data_dir, folder, img_path))
            axs[i, j].imshow(img, cmap='gray', interpolation='none')
            axs[i, j].set_title(folder)
            axs[i, j].axis('off')
    plt.show()

 # get the label of the first ten images and print out the outputs
first_ten_data, first_ten_label = utility.first_ten_output(greek_test_loader, network)
plot_predictions(image_dir,  9, 3, 3)

    # evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data, label in greek_test_loader:
            output = network(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

# print the accuracy
print(f"Accuracy on test set: {100 * correct / total:.2f}%")



       

  