
//
// Created by Avnish Patel
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "filter.h"

//#include "filter.cpp"
int main(int argc, char* argv[]) {
    cv::VideoCapture* capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame, currentFrame;
    cv::Mat grayscale;
 

    int brightness = 0;
    char param = ' ';

    cv:: namedWindow("Zoom", 1);
    int zoom = 100;
    cv::createTrackbar("Zoom", "Zoom", &zoom, 200);

    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        // see if there is a waiting paramstroke
        char key = cv::waitKey(10);
        if (key == ' ' || key == 'g' || key == 'h' || key == 'b' || key == 'x' || key == 'y' || key == 'm' || key == 'l' || key == 'c' || key == 'f') {
            param = key;
        }
        if (param == ' ') {
            // if user types ' ', display the original version of the image
            frame.copyTo(currentFrame);
        }
        else if (param == 'g') {
            cv::cvtColor(frame, currentFrame, cv::COLOR_BGR2GRAY);
        }
        else if (param == 'h') {
            greyscale(frame, currentFrame);
        }
        else if (param == 's') {
            cv::imwrite("capturedimage.jpg", frame);
        }
        else if (param == 'b') {
            blur5x5(frame, currentFrame);

        }
        else if (param == 'x') {
            // if user types 'x', display the sobelX version of the image
            cv::Mat resultFrame; // CV_16SC3
            sobelX3x3(frame, resultFrame);
            convertScaleAbs(resultFrame, currentFrame);
        }
        else if (param == 'y') {
            // if user types 'y', display the sobelY version of the image
            cv::Mat resultFrame; // CV_16SC3
            sobelY3x3(frame, resultFrame);
            convertScaleAbs(resultFrame, currentFrame);
        }
        else if (param == 'm') {
            cv::Mat sobelX, sobelY;
            sobelX3x3(frame, sobelX);
            sobelY3x3(frame, sobelY);
            cv::Mat resultFrame; // CV_16SC3
            magnitude(sobelX, sobelY, resultFrame);
            convertScaleAbs(resultFrame, currentFrame); //Scales, computes absolute values and converts the result to 8-bit.
        }
        else if (param == 'l') {
            // if user types 'l', display a blurred and quantized version of the image
            blurQuantize(frame, currentFrame, 15);
        }
        else if (param == 'c') {
            // if user types 'c', display a cartoon version of the image
            cartoon(frame, currentFrame, 15, 17);
        }
       
            // change the brightness of the video
            if (key == '1') {
                // increase brightness by 20
                brightness += 10;
            }
            else if (key == '2') {
                // decrease brightness by 20
                brightness -= 10;
            }
            else if (key == '3') {
                // reset to actual brightness
                brightness = 0;
            }
            currentFrame.convertTo(currentFrame, -1, 1, brightness);
            if (key == 'q') {
                break;
            }
            if (key == 's') {
                std::cout << "image saved" << std::endl;
                // ask for a meme for the saved processed image
                std::string photoMeme;
                std::cout << "Write your meme here: " << std::endl;
                std::getline(std::cin, photoMeme,'\n');
                cv::putText(currentFrame, photoMeme, cv::Point(refS.width / 10, refS.height / 1.3), cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(137, 160, 241),6);
                cv::imwrite("original.jpg", frame);
                cv::imwrite("processed.jpg", currentFrame);
            }
            cv::Mat zoomed;
            double fx = zoom / 100.0;
            double fy = zoom / 100.0;
            resize(frame, zoomed, cv::Size(), fx, fy,cv::INTER_LINEAR); //Dividing by 100 is used to convert the trackbar value 
                                                                   //to a decimal scaling factor, which is the format required by the resize function.

            cv::imshow("Zoom", zoomed);
            cv::imshow("Video", currentFrame);

        }

        delete capdev;
        //cv::destroyAllWindows();
        return 0;
    }
