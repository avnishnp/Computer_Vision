//
// Created by Avnish Patel
//

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "filter.h"

int greyscale(const cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i, j) = (src.at<cv::Vec3b>(i, j)[0] + src.at<cv::Vec3b>(i, j)[1] + src.at<cv::Vec3b>(i, j)[2]) / 3;
        }
    }
    return 0;
}

int blur5x5(const cv::Mat &src, cv::Mat &dst) {

// Create the 1x5 horizontal filter
    int hFilter[5] = { 1, 2, 4, 2, 1 };
    // Create the 5x1 vertical filter
    int vFilter[5] = { 1, 2, 4, 2, 1 };

    // Create temporary matrices for intermediate results
    cv::Mat temp(src.rows, src.cols, src.type());
    cv::Mat temp2(src.rows, src.cols, src.type());

    // Perform horizontal filtering
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int sum[3] = { 0 };
            for (int k = -2; k <= 2; k++) {
                if (j + k < 0 || j + k >= src.cols) continue;
                sum[0] += src.at<cv::Vec3b>(i, j + k)[0] * hFilter[k + 2];
                sum[1] += src.at<cv::Vec3b>(i, j + k)[1] * hFilter[k + 2];
                sum[2] += src.at<cv::Vec3b>(i, j + k)[2] * hFilter[k + 2];
            }
            temp.at<cv::Vec3b>(i, j)[0] = sum[0] / 10;
            temp.at<cv::Vec3b>(i, j)[1] = sum[1] / 10;
            temp.at<cv::Vec3b>(i, j)[2] = sum[2] / 10;
        }
    }

    // Perform vertical filtering
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int sum[3] = { 0 };
            for (int k = -2; k <= 2; k++) {
                if (i + k < 0 || i + k >= src.rows) continue;
                sum[0] += temp.at<cv::Vec3b>(i + k, j)[0] * vFilter[k + 2];
                sum[1] += temp.at<cv::Vec3b>(i + k, j)[1] * vFilter[k + 2];
                sum[2] += temp.at<cv::Vec3b>(i + k, j)[2] * vFilter[k + 2];
            }
            temp2.at<cv::Vec3b>(i, j)[0] = sum[0] / 10;
            temp2.at<cv::Vec3b>(i, j)[1] = sum[1] / 10;
            temp2.at<cv::Vec3b>(i, j)[2] = sum[2] / 10;
        }
    }

    // Copy the result to the destination matrix
    temp2.copyTo(dst);

    return 0;
}


int magnitude(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst) {
    // to use cv::sqrt, the input and output type need to be CV_32FC3
    cv::Mat sx32FC3, sy32FC3, dst32FC3;
    sx.convertTo(sx32FC3, CV_32FC3);
    sy.convertTo(sy32FC3, CV_32FC3);

    sqrt(sx32FC3.mul(sx32FC3) + sy32FC3.mul(sy32FC3), dst32FC3);

    // convert the dst type back to CV_16SC3
    dst32FC3.convertTo(dst, CV_16SC3);

    return 0;
}
int sobelX3x3(const cv::Mat &src, cv::Mat &dst)
{
    int filterX[3] = { 1, 0, -1 };

    dst = cv::Mat(src.rows, src.cols, CV_16SC3);

    // Applying horizontal filter
    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
                int sum[3] = { 0, 0, 0 };
                for (int x = -1; x <= 1; x++)
                {
                    cv::Vec3b wPixel = src.at<cv::Vec3b>(i, j + x);
                    sum[0] += wPixel[0] * filterX[x + 1];
                    sum[1] += wPixel[1] * filterX[x + 1];
                    sum[2] += wPixel[2] * filterX[x + 1];
                }
                dst.at<cv::Vec3s>(i, j)[0] = sum[0];
                dst.at<cv::Vec3s>(i, j)[1] = sum[1];
                dst.at<cv::Vec3s>(i, j)[2] = sum[2];
            }
        }
    }
    return 0;
}
int sobelY3x3(const cv::Mat &src, cv::Mat &dst)
{
    int filterY[3] = { -1, 0,1 };

    dst = cv::Mat(src.rows, src.cols, CV_16SC3);

    // Applying vertical filter
    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
                int sum[3] = { 0, 0, 0 };
                for (int x = -1; x <= 1; x++)
                {
                    cv::Vec3b wPixel = src.at<cv::Vec3b>(i + x, j);
                    sum[0] += wPixel[0] * filterY[x + 1];
                    sum[1] += wPixel[1] * filterY[x + 1];
                    sum[2] += wPixel[2] * filterY[x + 1];
                }
                dst.at<cv::Vec3s>(i, j)[0] = sum[0];
                dst.at<cv::Vec3s>(i, j)[1] = sum[1];
                dst.at<cv::Vec3s>(i, j)[2] = sum[2];
            }
        }
    }
    return 0;
}





int blurQuantize(const cv::Mat &src, cv::Mat &dst, int levels) {
    // blur the image
    blur5x5(src, dst);

    // quantize the image
    int b = 255 / levels;
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            for (int k = 0; k <= 2; k++) {
                dst.at<cv::Vec3b>(i, j)[k] = dst.at<cv::Vec3b>(i, j)[k] / b * b;
            }
        }
    }

    return 0;
}


int cartoon(const cv::Mat &src, cv::Mat &dst, int levels, int magThreshold) {
    // calculate the gradient magnitude and save the values into a new Mat
    cv::Mat sx, sy;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    cv::Mat mag;
    magnitude(sx, sy, mag);

    // blur and quantize the src image
    blurQuantize(src, dst, levels);

    // modify the blurred and quantized image
    // by setting to black any pixels with a gradient magnitude larger than the threshold
    for (int i = 0; i < mag.rows; i++) {
        for (int j = 0; j < mag.cols; j++) {
            if (mag.at<cv::Vec3s>(i, j)[0] > magThreshold || mag.at<cv::Vec3s>(i, j)[1] > magThreshold ||
                mag.at<cv::Vec3s>(i, j)[2] > magThreshold) {
                dst.at<cv::Vec3b>(i, j)[0] = 0;
                dst.at<cv::Vec3b>(i, j)[1] = 0;
                dst.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }

    return 0;
}




