//
// Avnish Patel
//

#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "process.h"
#include "csv_util.h"

using namespace cv;
using namespace std;

/*
 * The function first converts an image to grayscale,
 * and then set any pixel with a value less or equal to 130 as foreground, and other pixels as background
 *
 */
Mat threshold(Mat& image) {
    int THRESHOLD = 130;
    Mat processedImage, grayscale;
    processedImage = Mat(image.size(), CV_8UC1);

    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    for (int i = 0; i < grayscale.rows; i++) {
        for (int j = 0; j < grayscale.cols; j++) {
            if (grayscale.at<uchar>(i, j) <= THRESHOLD) {
                processedImage.at<uchar>(i, j) = 255;
            }
            else {
                processedImage.at<uchar>(i, j) = 0;
            }
        }
    }
    return processedImage;
}

/*
 * The function applies dilation on an image and then applies erosion on it
 *
 */
Mat cleanup(Mat& image) {
    Mat processedImage;
    const Mat kernel = getStructuringElement(MORPH_CROSS, Size(7, 7));
    morphologyEx(image, processedImage, MORPH_CLOSE, kernel);
    return processedImage;
}

/*
 * The function extracts the largest three regions of a given image, and writes the attributes into related Mats.
 *
 *  image: the given image
 *  labeledRegions: a Mat to store the label of each pixel
 * stats: a Mat to store the attributes of each labeled region
 *  centroids: a Mat to store the centroids of each labeled region
 *  topNLabels: a vector to store the labels of the largest 9 regions
 * @return an image of the largest three regions as a Mat
 */
Mat getRegions(Mat& image, Mat& labeledRegions, Mat& stats, Mat& centroids, vector<int>& topNLabels) {
    Mat processedImage;
    int nLabels = connectedComponentsWithStats(image, labeledRegions, stats, centroids);

    // save all region areas into a vector and sort the area descending
    Mat areas = Mat::zeros(1, nLabels - 1, CV_32S);
    Mat sortedIdx;
    for (int i = 1; i < nLabels; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        areas.at<int>(i - 1) = area;
    }
    if (areas.cols > 0) {
        sortIdx(areas, sortedIdx, SORT_EVERY_ROW + SORT_DESCENDING);
    }

    vector<Vec3b> colors(nLabels, Vec3b(0, 0, 0)); // label to color mapping

    int N = 9; // only take the largest 3 non-background regions
    N = (N < sortedIdx.cols) ? N : sortedIdx.cols;
    int THRESHOLD = 50; // any region area less than 5,0 will be ignored
    for (int i = 0; i < N; i++) {
        int label = sortedIdx.at<int>(i) + 1;
        if (stats.at<int>(label, CC_STAT_AREA) > THRESHOLD) {
            colors[label] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
            topNLabels.push_back(label);
        }
    }

    processedImage = Mat::zeros(labeledRegions.size(), CV_8UC3);
    for (int i = 0; i < processedImage.rows; i++) {
        for (int j = 0; j < processedImage.cols; j++) {
            int label = labeledRegions.at<int>(i, j);
            processedImage.at<Vec3b>(i, j) = colors[label];
        }
    }
    return processedImage;
}

/*
 * The function computes the rotated bounding box of a given region
 *
 *  region: the given region
 *  x: the x-axis value of the centroid of the region
 *  y: the y-axis value of the centroid of the region
 *  alpha: the angle between the x-axis and least central x-axis
 *  a rotated bounding box of the given region
 */
RotatedRect getBoundingBox(Mat& region, double x, double y, double alpha) {
    int maxX = INT_MIN, minX = INT_MAX, maxY = INT_MIN, minY = INT_MAX;
    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            if (region.at<uchar>(i, j) == 255) {
                int projectedX = (i - x) * cos(alpha) + (j - y) * sin(alpha);
                int projectedY = -(i - x) * sin(alpha) + (j - y) * cos(alpha);
                maxX = max(maxX, projectedX);
                minX = min(minX, projectedX);
                maxY = max(maxY, projectedY);
                minY = min(minY, projectedY);
            }
        }
    }
    int lengthX = maxX - minX;
    int lengthY = maxY - minY;

    Point centroid = Point(x, y);
    Size size = Size(lengthX, lengthY);

    return RotatedRect(centroid, size, alpha * 180.0 / CV_PI);
}

/*
 * This function draws a line of 100 pixels given a starting point and an angle
 *
 * image: the image where the line is drawn
 *  x: the value of x-axis of the starting point
 *  y: the value of y-axis of the starting point
 *  alpha: the given angle
 *  color: the color of the line to be drawn
 */
void drawLine(Mat& image, double x, double y, double alpha, Scalar color) {
    double length = 100.0;
    double edge1 = length * sin(alpha);
    double edge2 = sqrt(length * length - edge1 * edge1);
    double xPrime = x + edge2, yPrime = y + edge1;

    arrowedLine(image, Point(x, y), Point(xPrime, yPrime), color, 3);
}

/*
 * This function draws a rectangle on a given image
 *
 *  image: the given image
 *  boundingBox: the rectangle to be drawn
 *  color: the color of the rectangle
 */
void drawBoundingBox(Mat& image, RotatedRect boundingBox, Scalar color) {
    Point2f rect_points[4];
    boundingBox.points(rect_points);
    for (int i = 0; i < 4; i++) {
        line(image, rect_points[i], rect_points[(i + 1) % 4], color, 3);
    }
}

/*
 * This function calculates the HU Moments according to the given central moments
 *
 * mo: the given Moments contains the central moments
 *  huMoments: a vector to store the 7 attributes of HU Moments
 */
void calcHuMoments(Moments mo, vector<double>& huMoments) {
    double hu[7]; // HuMoments require the parameter type to be double[]
    HuMoments(mo, hu);

    // covert array to vector
    for (double d : hu) {
        huMoments.push_back(d);
    }
    return;
}

/*
 * This function calculates the normalized Euclidean distance between two vectors
 *
 *  feature1: the first vector
 * feature2: the second vector
 *
 * @return the normalized distance as a double
 */
double euclideanDistance(vector<double> features1, vector<double> features2) {
    double sum1 = 0, sum2 = 0, sumDifference = 0.0;
    for (int i = 0; i < features1.size(); i++) {
        sumDifference += (features1[i] - features2[i]) * (features1[i] - features2[i]);
        sum1 += features1[i] * features1[i];
        sum2 += features2[i] * features2[i];
    }
    return sqrt(sumDifference) / (sqrt(sum1) + sqrt(sum2));
}

/*
 * Given some data and a feature vector, this function gets the class name of the given feature vector
 * Infers based on the nearest neighbor, and use normalized euclidean distance as distance metric
 *
 * featureVectors: a vector to store the feature vectors of known objects
 *  classNames: a vector to store the class names of known objects
 * currentFeature: the feature vector of the object needed to be inferred
 *  the inferred class name as a string
 */
string classifier(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature) {
    double THRESHOLD = 0.15;
    double distance = DBL_MAX;
    string className = " ";
    for (int i = 0; i < featureVectors.size(); i++) { // loop the known features to get the closed one
        vector<double> dbFeature = featureVectors[i];
        string dbClassName = classNames[i];
        double curDistance = euclideanDistance(dbFeature, currentFeature);
        if (curDistance < distance && curDistance < THRESHOLD) {
            className = dbClassName;
            distance = curDistance;
        }
    }
    return className;
}

/*
 * Given some data and a feature vector, this function gets the name of the given feature vector
 * Infers based on K-Nearest-Neighbor, and use normalized euclidean distance as distance metric
 *
 *  featureVectors: a vector to store the feature vectors of known objects
 * classNames: a vector to store the class names of known objects
 * currentFeature: the feature vector of the object needed to be inferred
 *  K: the k value in KNN
 *  the inferred class name as a string
 */
string classifierKNN(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, int K) {
    double THRESHOLD = 0.15;
    // compute the distances of current feature vector with all the feature vectors in DB
    vector<double> distances;
    for (int i = 0; i < featureVectors.size(); i++) {
        vector<double> dbFeature = featureVectors[i];
        double distance = euclideanDistance(dbFeature, currentFeature);
        if (distance < THRESHOLD) {
            distances.push_back(distance);
        }
    }

    string className = " ";
    if (distances.size() > 0) {
        // sort the distances in ascending order
        vector<int> sortedIdx;
        sortIdx(distances, sortedIdx, SORT_EVERY_ROW + SORT_ASCENDING);

        // get the first K class name, and count the number of each name
        vector<string> firstKNames;
        int s = sortedIdx.size();
        map<string, int> nameCount;
        int range = min(s, K); // if less than K classnames, get all of them
        for (int i = 0; i < range; i++) {
            string name = classNames[sortedIdx[i]];
            if (nameCount.find(name) != nameCount.end()) {
                nameCount[name]++;
            }
            else {
                nameCount[name] = 1;
            }
        }

        // get the class name that appear the most times in the K nearest neighbors
        int count = 0;
        for (map<string, int>::iterator it = nameCount.begin(); it != nameCount.end(); it++) {
            if (it->second > count) {
                className = it->first;
                count = it->second;
            }
        }
    }
    return className;
}

/*
 * This function returns the corresponding class name given a code
 *
 *  c: the code for each class name
 *  the class name as a string
 */
string getClassName(char c) {
    std::map<char, string> myMap{
           
            {'c', "charger1"}, {'s', "charger2"} , {'k', "knife"},{'m',"mouse"},{'u',"pouch"},{'p',"pen"},{'g',"box"},{'o',"cover"},{'n',"spoon"},{'e',"airpod"}
    };
    return myMap[c];
}