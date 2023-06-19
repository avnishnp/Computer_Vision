//
// Avnish Patel
//

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;

void bodyDetection(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    // Detect bodies
    std::vector<Rect> bodies;
    face_cascade.detectMultiScale(frame_gray, bodies);

    for (size_t i = 0; i < bodies.size(); i++)
    {
        //Point center(bodies[i].x + bodies[i].width / 2, bodies[i].y + bodies[i].height / 2);
       // ellipse(frame, center, Size(bodies[i].width / 2, bodies[i].height / 2), 0, 0, 360, Scalar(0, 255, 255), 6);
        //Mat faceROI = frame_gray(bodies[i]);

        rectangle(frame, bodies[i], Scalar(0, 255, 255), 6);
    }

    imshow("Live Body Detection", frame);
}


int main(int argc, const char** argv)
{

    // Load the pre trained haar cascade classifier

    //string faceClassifier = "C:/Users/kemik/OneDrive/Skrivebord/haarcascade_frontalface_default.xml";
    string bodyClassifier = "E:/Project3_haarcascade/haarcascade_fullbody.xml";

    if (!face_cascade.load(bodyClassifier))
    {
        cout << "Could not load the classifier";
        return -1;
    };

    cout << "Classifier Loaded!" << endl;

    // Read the video stream from camera
    VideoCapture capture("C:/Users/Avnish/Downloads/v1.mp4");

    if (!capture.isOpened())
    {
        cout << "Could not open video capture";
        return -1;
    }

    // Define the codec and create VideoWriter object
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = capture.get(CAP_PROP_FPS);
    Size frame_size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter output("output.mp4", codec, fps, frame_size);

    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "No frame captured from camera";
            break;
        }

        // Apply the face detection with the haar cascade classifier
        bodyDetection(frame);

        if (waitKey(10) == 'q')
        {
            break; // Terminate program if q pressed
        }
    }
    capture.release();
    output.release();
    return 0;
}