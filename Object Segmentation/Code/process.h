//
// Avnish Patel
//

#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
 * The function first converts an image to grayscale,
 * and then set any pixel with a value less or equal to 130 as foreground, and other pixels as background
 *

 */
Mat threshold(Mat& image);

/*
 * The function applies dilation on an image and then applies erosion on it
 *

 */
Mat cleanup(Mat& image);

/*
 * The function extracts the largest three regions of a given image, and writes the attributes into related Mats.
 *
Mat getRegions(Mat& image, Mat& labeledRegions, Mat& stats, Mat& centroids, vector<int>& topNLabels);

/*
 * The function computes the rotated bounding box of a given region
 *
 */
RotatedRect getBoundingBox(Mat& region, double x, double y, double alpha);

/*
 * This function draws a line of 100 pixels given a starting point and an angle
 *
 */
void drawLine(Mat& image, double x, double y, double alpha, Scalar color);

/*
 * This function draws a rectangle on a given image
 *
 */
void drawBoundingBox(Mat& image, RotatedRect boundingBox, Scalar color);

/*
 * This function calculates the HU Moments according to the given central moments
 *
 */
void calcHuMoments(Moments mo, vector<double>& huMoments);

/*
 * This function calculates the normalized Euclidean distance between two vectors

 */
double euclideanDistance(vector<double> features1, vector<double> features2);

/*
 * Given some data and a feature vector, this function gets the class name of the given feature vector
 * Infers based on the nearest neighbor, and use normalized euclidean distance as distance metric
 *
 */
string classifier(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature);

/*
 * Given some data and a feature vector, this function gets the name of the given feature vector
 * Infers based on K-Nearest-Neighbor, and use normalized euclidean distance as distance metric
 *
 */
string classifierKNN(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, int K);

/*
 * This function returns the corresponding class name given a code
 */
string getClassName(char c);