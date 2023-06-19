
# 
# Avnish Patel
#


import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
from mnist import MyNetwork
import utility

#  the code is first analyzing and plotting the filters of the first and second convolutional layers of two pre-trained models (model and sub_model),
#  and then applying these filters to the first image from the MNIST training dataset and visualizing the filtered output.

'''
Load the model trained by mnist.py
Plot the 10 filters in the first convolution layer of the model
Load the MNIST training data
Apply the 10 filters to the first image in the training dataset
Plot the 10 filters and 10 filtered images
'''

def main(argv):
    # load and print the model
    model = torch.load('model.pth')
    print(model)

    # for conv1 layer, print the filter weights and the shape
    # plot the 10 filters
    filters = utility.plot_filters(model.conv1, 10, 3, 4)

    # apply the 10 filters to the first training example image
    # load training data
    train_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))
    #  get the first image
    #  Extract the first image and its corresponding label from the MNIST training dataset using the next function and
    #  the iter function to create an iterator over the dataset.
    first_image, first_label = next(iter(train_loader))

    # Extract a 2D numpy array from the first image tensor using the squeeze function to remove the color channel dimension,
    #  and then transpose the array so that the channel dimension is the last dimension. The resulting array is stored in squeezed_image.


    # first_image is a tensor with shape (batch_size, channels, height, width)
    # by squeeze we get (batch_size, height, width)
    # The numpy() function is then called to convert this tensor to a NumPy array.
    # Finally, transpose() is called to change the order of the dimensions from (batch_size, height, width) to (height, width, batch_size)
    #  so that it can be plotted using Matplotlib.
    squeezed_image = np.transpose(torch.squeeze(first_image, 1).numpy(), (1, 2, 0))

    # plot the first images filtered by the 10 filters from layer 1
    utility.plot_filtered_images(filters, squeezed_image, 10, 20, 5, 4)


if __name__ == "__main__":
    main(sys.argv)