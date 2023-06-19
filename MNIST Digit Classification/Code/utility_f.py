
# 
# Avnish Patel
#

# This file contains some pre-defined hyper-parameters,
# model definition, and helper functions definition

# import statements
import csv
import cv2
from collections import Counter

# import cv2
import torch
from numpy import linalg
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# define hyper-parameters
N_EPOCHS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10

'''
The function plots images
 data: the images to be plotted
 row: number of rows in the plot
 col: number of columns in the plot
'''
def plot_images(data, row, col):
    examples = enumerate(data)
    batch_idx, (example_data, example_targets) = next(examples)
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

'''
The function plots curves of the training loses and testing losses
 train_counter: array of train counter
 train_losses: array of train losses
 test_counter: array of test counter
 test_losses: array of test losses
'''
def plot_curve(train_counter, train_losses, test_counter, test_losses):
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

'''
The function apply model on dataset and get the first 10 data and the labels
 data: the testing data
 model: the model used
 first_ten_data: array contains the first 10 data
 first_ten_label: array contains the label of the first 10 data
'''
def first_ten_output(data, model):
    first_ten_data = []
    first_ten_label = []

    count = 0
    for data, target in data:
        if count < 100:
            # Here, the if statement ensures that only the first ten data samples are processed.
            #  Inside the loop, the input data is first squeezed to remove any unnecessary dimensions and then converted to a numpy array
            squeeze_data = np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0))
            first_ten_data.append(squeeze_data)
            # Here, the with torch.no_grad() context ensures that PyTorch does not keep track of the computation graph for the output predictions, which saves memory
            with torch.no_grad():
                output = model(data)
                print(f'{count + 1} - output: {output}')
                print(f'{count + 1} - index of the max output value: {output.argmax().item()}')
                # The output predictions are printed along with the index of the maximum output value and the corresponding prediction label.
                label = output.data.max(1, keepdim=True)[1][0].item()
                print(f'{count + 1} - prediction label: {label}')
                first_ten_label.append(label)
                count += 1

    return first_ten_data, first_ten_label


'''
The function plots the image and their prediction values
 data_set: the image to be plotted
 label_set: the labels of the dataset
 total: total number of data to be plotted
 row: number of rows in the plot
 col: number of columns in the plot
'''
def plot_prediction(data_set, label_set, total, row, col):
    for i in range(total):
        plt.subplot(row, col, i + 1)
        plt.tight_layout()
        plt.imshow(data_set[i], cmap='gray', interpolation='none')
        plt.title('Pred: {}'.format(label_set[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

'''
the plot_filters function is used to visualize the filters of a convolutional layer
 conv: the convolutation layer from a model which contains the filters to be plotted
 total: total number of filters to be plotted
 row: number of rows in the plot
 col: number of columns in the plot
 filters:; array of all the filters plotted
'''
def plot_filters(conv, total, row, col):
    filters = []
    with torch.no_grad():
        for i in range(total):
            plt.subplot(row, col, i + 1)
            plt.tight_layout()
            # Extract the i-th filter from the convolutional layer
            curr_filter = conv.weight[i, 0]
            # Append the current filter to the filters list.
            filters.append(curr_filter)
            print(f'filter {i + 1}')
            print(curr_filter)
            print(curr_filter.shape)
            print('\n')
            plt.imshow(curr_filter)
            plt.title(f'Filter {i + 1}')
            plt.xticks([])
            plt.yticks([])
        plt.show()
    return filters


'''
 plot_filtered_images function is used to visualize the output of a convolutional layer given an input image.
 filters: the filters to be plotted
 image: the image to be filtered
 n: the total number of filters
 total: total number of images in the plot
 row: number of rows in the plot
 col: number of columns in the plot
'''
def plot_filtered_images(filters, image, n, total, row, col):
    with torch.no_grad():
        items = []
        for i in range(n):
            items.append(filters[i])
            #  applies each filter to the input image using the filter2D function from OpenCV, which performs a 2D convolution between the filter and the image.
            filtered_image = cv2.filter2D(np.array(image), ddepth=-1, kernel=np.array(filters[i]))
            items.append(filtered_image)
        for i in range(total):
            plt.subplot(row, col, i + 1)
            plt.tight_layout()
            plt.imshow(items[i])
            plt.xticks([])
            plt.yticks([])
        plt.show()