# Project: LiDAR-Camera Fusion and Obstacle Detection

This project demonstrates the process of fusing LiDAR point clouds with camera images to enhance object detection using YOLOv8. The fusion process projects LiDAR points onto image frames and detects obstacles via a deep learning-based model.

## 📂 File Descriptions

### `early_fusion.py`
This script serves as the entry point for the project. It reads input images, point clouds, and calibration files, and then processes them through the pipeline to detect objects using YOLO and fuse LiDAR points with camera images.

### `detect_obstacle.py`
Contains the implementation of the YOLOv8-based object detection class `YoloOD`. It supports both tiny and normal models for detecting objects in the provided camera frames.

### `transform_lidar_camera.py`
Implements the `LiDAR2Camera` class responsible for transforming LiDAR points to the camera coordinate system, projecting them onto images, and filtering outliers. It also includes functions for performing fusion and calculating distance to detected objects.

### `gen_video.py`
Creates a video from a series of images and point clouds by applying the LiDAR-Camera fusion and YOLO-based object detection pipeline to each frame. It saves the processed frames and generates a video as output.

## 🔧 Requirements
- Python 3.8+
- OpenCV
- Open3D
- Ultralytics YOLOv8
- NumPy
- Matplotlib

## 🚀 Usage
- Run `early_fusion.py` to process a single frame.
- Run `gen_video.py` to generate a video from a series of frames.

## 📂 Output
- Processed frames saved to the `output_images` directory.
- Generated video saved as `out_4.mp4`.

---
## 🔍 Sample Outputs

### Output Example 1
![Output1](output_images/000004.png)

### Output Example 2
![Output 2](output_images/000000.png)