
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

const double PI = 3.14;
#include <cmath>

const int kernel_size = 9;
const double sigma = 2;

double gaussian_kernel[kernel_size][kernel_size];
double laplacian_kernel[kernel_size][kernel_size];

void create_gaussian_kernel() {
    double sum = 0.0;
    int center = kernel_size / 2;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            double x = i - center;
            double y = j - center;
            gaussian_kernel[i][j] = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
            sum += gaussian_kernel[i][j];
        }
    }
    // normalize the kernel
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            gaussian_kernel[i][j] /= sum;
        }
    }
}

void create_laplacian_kernel() {
    double sum = 0.0;
    int center = kernel_size / 2;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            double x = i - center;
            double y = j - center;
            laplacian_kernel[i][j] = -(x * x + y * y - 2 * sigma * sigma) / (sigma * sigma * sigma * sigma) * gaussian_kernel[i][j];
            sum += laplacian_kernel[i][j];
        }
    }
    // normalize the kernel
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            laplacian_kernel[i][j] /= sum;
        }
    }
}
void convolution(const cv::Mat &src, cv::Mat &dst, double kernel[kernel_size][kernel_size], int kernel_size) {
    int center = kernel_size / 2;
    for (int i = center; i < src.rows - center; i++) {
        for (int j = center; j < src.cols - center; j++) {
            double sum = 0.0;
            for (int k = -center; k <= center; k++) {
                for (int l = -center; l <= center; l++) {
                    sum += src.at<uchar>(i + k, j + l) * kernel[k + center][l + center];
                }
            }
            dst.at<uchar>(i, j) = sum;
        }
    }
}
int main(int argc, char* argv[]) {
    create_gaussian_kernel();
    create_laplacian_kernel();
    cv::Mat img = cv::imread("C:/Users/Avnish/OneDrive/Desktop/zebra.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat filtered_img(img.rows, img.cols, CV_8UC1);
    cv::imshow("img1", img);
    convolution(img, filtered_img, gaussian_kernel, kernel_size);
    convolution(filtered_img, filtered_img, laplacian_kernel, kernel_size);
    cv::imshow("img2", filtered_img);

    cv::waitKey(0);
    return 0;
}