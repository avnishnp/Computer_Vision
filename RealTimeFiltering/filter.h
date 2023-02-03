//
// Created by Avnish Patel
//

#pragma once
#include <opencv2/opencv.hpp>
int greyscale(const cv::Mat &src, cv::Mat &dst);
 int blur5x5(const cv::Mat &src, cv::Mat &dst);
 int sobelX3x3(const cv::Mat &src, cv::Mat &dst);
 int sobelY3x3(const cv::Mat &src, cv::Mat &dst);
 int blurQuantize(const cv::Mat &src, cv::Mat &dst, int levels);
int magnitude(const cv::Mat &sx, const cv::Mat &sy, cv::Mat &dst);
int cartoon(const cv::Mat &src, cv::Mat &dst, int levels, int magThreshold);

