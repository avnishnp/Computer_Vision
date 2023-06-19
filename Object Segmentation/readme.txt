//
Avnish Patel
//

Links/URLs to any videos you created and want to submit as part of your report
No

What operating system and IDE you used to run and compile your code.
Windows 11 , Visual Studio

Instructions for running your executables.

I have used "iriun" IP based webcam from mobile. If you use anything else , you will have to change the argument of video capture object.

The executable takes two input. The first is the path to the csv file store the class name and feature vector for each known object.
The second is the classifier type ('n' for the nearest neighbor, 'k' for KNN). 

Example is "obj_db n" in command line argument of Visual Studio

 Instructions for testing any extensions you completed.
For the haarcascade extensions , use the "haarcascade_frontalface_alt2.xml" file for loading the classifier.

Whether you are using any time travel days and how many.
Yes, 2 days

Other important info : 
I have named the labels as following for the classifier

{'c', "charger1"}, {'s', "charger2"} , {'k', "knife"},{'m',"mouse"},{'u',"pouch"},
{'p',"pen"},{'g',"box"},{'o',"cover"},{'n',"spoon"},{'e',"airpod"}
 
The haarcascade extension output is in the form of a video "harrcascade_video.mp4"
The Task 9 "Capture a demo of your system working" video is "classify_video.mp4"

Overall the files for the tasks are :
main.cpp
process.cpp
process.h
csv_util.cpp
csv_util.h
classify_video

Files for haarcascade are :
haar_cascade.cpp
haarcascade_frontalface_alt2
harrcascade_video
