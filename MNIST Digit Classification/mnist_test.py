# import statements
import utility
from mnist import MyNetwork

import sys
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

# set output precision to 2
torch.set_printoptions(precision=2)


'''
Load the model trained and saved by MNISTRecognition.py
Load the MNIST test data
Get the first ten test image and their predictions, plot the results
Load the custom 0 - 9 digits, apply the trained model
Plot the custom digits and their predictions
'''
def main(argv):
    # load the model
    model = torch.load('model.pth')
    model.eval()

    # load test data
    test_loader = DataLoader(
        torchvision.datasets.MNIST('data2', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))

    # get the label of the first ten images and print out the outputs
    first_ten_data, first_ten_label = utility.first_ten_output(test_loader, model)

    # plot the predictions for the first ten images
    utility.plot_prediction(first_ten_data, first_ten_label, 9, 3, 3)

    # load custom digit data, apply the model, and plot the ten results
    image_dir = "E:/Computer_Vision/Project6/0_9"
    custom_images = datasets.ImageFolder(image_dir,
                                         transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                       transforms.Grayscale(),
                                                                       transforms.functional.invert,
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))]))
    custom_loader = DataLoader(custom_images)

    first_ten_custom_data, first_ten_custom_label = utility.first_ten_output(custom_loader, model)
    utility.plot_prediction(first_ten_custom_data, first_ten_custom_label, 10, 3, 4)

    return


if __name__ == "__main__":
    main(sys.argv)