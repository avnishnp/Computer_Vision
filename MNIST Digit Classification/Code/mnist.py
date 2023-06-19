
# 
# Avnish Patel
#

# import statements
import utility

import ssl
import sys
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torch import nn


# define hyper-parameters
N_EPOCHS = 100
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10


# model definition

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

'''
The function trains the model and save the model and optimizer
 epoch: number of epochs of the training process
 model: the model to be trained
 optimizer: the optimizer used when training
 train_loader: the training data
 train_losses: array to record the train losses
 train_counter: array to record the train counter
'''    
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
Load the MNIST training and testing dataset
Plot the first 6 images in the training dataset
Train and test the model, plot the training curve
Save the model and its state dict
'''
def main(argv):
    # make the network code repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # load test and training data
    train_loader = DataLoader(
        torchvision.datasets.MNIST('Mnist', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_loader = DataLoader(
        torchvision.datasets.MNIST('Mnist', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=BATCH_SIZE_TEST, shuffle=True)

    # plot the first 6 example digits
    utility.plot_images(train_loader, 2, 3)

    # initialize the network and the optimizer
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=utility.LEARNING_RATE,
                          momentum=utility.MOMENTUM)
    train_losses = []
    train_counter = []
    test_losses = []

    # The test counter is used to keep track of the number of examples seen during testing

    # The range(utility.N_EPOCHS + 1) specifies the number of epochs plus one because the model also calculates the initial test loss before any training has occurred.
    # number of samples in one epoch by (i * len(train_loader.dataset)
    test_counter = [i * len(train_loader.dataset) for i in range(utility.N_EPOCHS + 1)]

    # run the training
    test(network, test_loader, test_losses)
    # This loop is responsible for training and evaluating the performance of the neural network for multiple epochs.
    for epoch in range(1, utility.N_EPOCHS + 1):
        train(epoch, network, optimizer, train_loader, train_losses, train_counter)
        test(network, test_loader, test_losses)

    # plot training curve
    utility.plot_curve(train_counter, train_losses, test_counter, test_losses)

    # save the model
    torch.save(network, 'model.pth')
    torch.save(network.state_dict(), 'model_state_dict.pth')

    return


if __name__ == "__main__":
    main(sys.argv)