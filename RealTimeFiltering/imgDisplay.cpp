
//
// Created by Avnish Patel
//
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    // read an image
    Mat image;
    image = imread("C:/Users/Avnish/OneDrive/Desktop/image.jpg");

    // check image data
    if (!image.data) {
        cout << "No  data\n";
        return -1;
    }

    // display image
    namedWindow(" Task 1", WINDOW_AUTOSIZE);
    imshow("Task 1", image);

    // check for keypress.
    while (true) {
        char key = waitKey(10);
        // if user types 'q', quit
        if (key == 'q') {
            break;
        }
    }

    // destroy all the windows created
    destroyAllWindows();
    return 0;
}