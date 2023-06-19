# 
# Avnish Patel
#

# import statements
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from mnist import MyNetwork
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import utility
torch.backends.cudnn.enabled = False

# Load pre-trained model

network =torch.load('model.pth')
network.eval()
for param in network.parameters():
    param.requires_grad = False


newfc_2=nn.Linear(50,3)
network.fc2=newfc_2
print(network)

# Greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


image_dir = 'E:/Computer_Vision/Project6/greek'
    # It applies some image transformations using transforms.Compose, including resizing the images to (28, 28), converting them to grayscale,
    #  inverting them, converting them to PyTorch tensors, and normalizing them.
greek_test_loader = datasets.ImageFolder(image_dir,
                                 transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                transforms.Grayscale(),
                                                                 transforms.functional.invert,
                                                                transforms.ToTensor(),
                                                               transforms.Normalize((0.1307,), (0.3081,))]))

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(network.fc2.parameters(), lr=0.001, momentum=0.5)
# optimizer = torch.optim.SGD(net.mnist_net.fc2.parameters(), lr=0.01,momentum=0.5)
optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.5)


# Train the network
batch_size = 5
shuffle = True
greek_test_loader = DataLoader(greek_test_loader, batch_size=batch_size, shuffle=shuffle)

num_epochs = 100
losses = []
total_batches = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(greek_test_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = network(inputs)
         # Convert labels to a tensor
        labels = labels.long()
        loss = criterion(outputs, labels)
        # loss = torch.autograd.Variable(loss, requires_grad=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

         # Store the loss after every batch
        losses.append(loss.item())
        total_batches += 1

    print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / (i + 1)))

# Plot the graph
plt.plot(range(total_batches), losses)
plt.xlabel('Number of training examples')
plt.ylabel('Loss')
plt.show()

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
       
 # get the label of the first ten images and print out the outputs
first_ten_data, first_ten_label = utility.first_ten_output(greek_test_loader, network)
utility.plot_prediction(first_ten_data, first_ten_label, 36, 6, 6)

  