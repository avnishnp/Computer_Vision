
# 
# Avnish Patel
#

# import statements
import utility_f

import ssl
import sys
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

# define hyper-parameters
N_EPOCHS = 20
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10
batch_size_test = 64

# model definition

# A deep network with the following layers:
# A convolution layer with 10 5x5 filters
# A max pooling layer with a 2x2 window and a ReLU function applied.
# A convolution layer with 20 5x5 filters
# A dropout layer with a 0.5 dropout rate (50%)
# A max pooling layer with a 2x2 window and a ReLU function applied
# A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
# A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
import torch.nn as nn
from torch.nn.functional import relu, tanh, sigmoid
from scipy.stats import uniform
import random


class MyNetwork(nn.Module):
    def __init__(self, num_of_conv, conv_filter_size, dropout_rate):
        super(MyNetwork, self).__init__()
        self.input_size = 28 # input image size is 28x28
        self.num_of_conv = num_of_conv
        self.conv1 = nn.Conv2d(1, 10, kernel_size=conv_filter_size, padding='same')
        self.conv2 = nn.Conv2d(10, 20, kernel_size=conv_filter_size, padding='same')
        self.conv = nn.Conv2d(20, 20, kernel_size=conv_filter_size, padding='same')
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(self.get_fc1_input_size(), 50)
        self.fc2 = nn.Linear(50, 10)

    '''
    The function gets the input size for the first fully connected layer
    '''
    def get_fc1_input_size(self):
        fc1_input_size = self.input_size / 2
        fc1_input_size = fc1_input_size / 2
        fc1_input_size = fc1_input_size * fc1_input_size * 20
        return int(fc1_input_size)

    # define forward pass
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        for i in range(self.num_of_conv):
            x = F.relu(self.conv(x))
        # x = x.view(-1, )
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, 1)


def train(epoch, model, optimizer, train_loader, train_losses, train_counter):
    # sets the model in training mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # The optimizer gradients are reset to zero
        optimizer.zero_grad()
        # The input data is passed through the model to generate output predictions
        output = model(data)
        #  The negative log likelihood loss between the output predictions and the target labels
        loss = F.nll_loss(output, target)
        #  The gradients of the loss with respect to the model parameters are computed
        loss.backward()
        # update the model parameters using the computed gradients
        optimizer.step()
        # the current loss and training step will be printed out depending on the value of LOG_INTERVAL.
        # If the current batch index is evenly divisible by LOG_INTERVAL, then the statement inside the if block is executed.
        if batch_idx % LOG_INTERVAL == 0:
            # The current loss is appended to the train_losses
            train_losses.append(loss.item())
            # each batch contains 64 training examples.
            # len(train_loader.dataset) returns the total number of training examples in the train_loader.
            #  Multiplying this by (epoch - 1) gives the total number of examples that have been processed in previous epochs.
            
            # calculates the total number of training examples that have been processed so far in the current epoch, 
            # as well as any examples processed in previous epochs.

            # The train_counter list keeps track of the total number of training examples processed up to the end of each batch during training.
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            # torch.save(model.state_dict(), 'results/model.pth')
            # torch.save(optimizer.state_dict(), 'results/optimizer.pth')


'''
The function tests the model and print the accuracy information
 model: the model to be tested
 test_loader: the test data
 test_losses: array to record test losses
'''

def test(model, test_loader, test_losses):
    # Set the model to evaluation mode 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            # The F.nll_loss() function computes the negative log-likelihood loss between the output predictions and the target labels.
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # The pred variable stores the index of the maximum value in the output predictions, which corresponds to the predicted label.
            pred = output.data.max(1, keepdim=True)[1]
            # The correct variable is updated by comparing the predicted labels with the target labels.
            correct += pred.eq(target.data.view_as(pred)).sum()
     # Compute the test loss and accuracy and update the test_losses list:
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    # The accuracy is computed by dividing the number of correct predictions by the total number of samples in the test dataset
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


'''
The function loads training and test data, initializes a network, trains the network.
The function prints the model accuracy and plots the training and testing losses.
 num_epochs: number of epochs of the training process
 batch_size_train: the batch size of the traning data
 num_of_conv: the number of convolution layers in the model
 conv_filter_size: the filter size in the convolution layers
 dropout_rate: the dropout rate of the model
'''
def experiment(num_epochs, batch_size_train, num_of_conv, conv_filter_size, dropout_rate):
    # load test and training data
    train_loader = DataLoader(
        torchvision.datasets.FashionMNIST('experiment_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train)

    test_loader = DataLoader(
        torchvision.datasets.FashionMNIST('experiment_data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test)

    # initialize the network and the optimizer
    network = MyNetwork(num_of_conv, conv_filter_size, dropout_rate)
    optimizer = optim.SGD(network.parameters(), lr=utility_f.LEARNING_RATE,
                          momentum=utility_f.MOMENTUM)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(num_epochs + 1)]

    # run the training
    utility_f.test(network, test_loader, test_losses)
    for epoch in range(1, num_epochs + 1):
        train(epoch, network, optimizer, train_loader, train_losses, train_counter)
        test(network, test_loader, test_losses)

    # plot training curve
    plot_curve(train_counter, train_losses, test_counter, test_losses)





'''
The function plots curves of the training loses and testing losses
@:parameter train_counter: array of train counter
@:parameter train_losses: array of train losses
@:parameter test_counter: array of test counter
@:parameter test_losses: array of test losses
'''
def plot_curve(train_counter, train_losses, test_counter, test_losses):
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')

'''
Run 72 experiments using experiment() by modifying the parameters, and display the results
epoch sizes: 3, 5
training batch sizes: 64, 128
the number of convolution layers: add an additional 1 - 3 convolution layers
convolution layer filter size: 3, 5, 7
dropout rate: 0.3, 0.5
'''
def main():
    for num_epochs in [3, 5]:
        for batch_size_train in [64, 128]:
            for num_of_conv in range(1, 4):
                for conv_filter_size in [3, 5, 7]:
                    for dropout_rate in [0.3, 0.5]:
                        print('______________________________')
                        print(f'Number of Epochs: {num_epochs}')
                        print(f'Train Batch Size: {batch_size_train}')
                        print(f'Number of Convolution Layer: {num_of_conv}')
                        print(f'Convolution Filter Size: {conv_filter_size}')
                        print(f'Dropout Rate: {dropout_rate}')
                        print('______________________________')
                        experiment(num_epochs, batch_size_train, num_of_conv, conv_filter_size, dropout_rate)


if __name__ == "__main__":
    main()