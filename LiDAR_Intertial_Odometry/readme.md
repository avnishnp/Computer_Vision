# LiDAR-Inertial Odometry (LIO) System

This project implements a **Loosely Coupled LiDAR-Inertial Odometry (LIO)** system using the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/). It combines **IMU preintegration** (via [PyPose](https://pypose.org/)) and **LiDAR scan matching** (via [Open3D](http://www.open3d.org/)) to estimate accurate 6-DoF poses over time.

## 📌 Features

- IMU-based motion propagation using PyPose's IMUPreintegrator.
- LiDAR-based pose correction using Generalized ICP (GICP).
- KITTI dataset integration and calibration handling.
- Visualization of estimated and ground-truth trajectories.
- Configurable YAML-based setup for easy parameter tuning.

## 📁 Project Structure

├── kitti/ \
│ ├── dataloader.py # Custom dataloader for KITTI \
│ └── calib.py # Handles calibration matrices \
├── utils.py # Helper functions (e.g., visualization, point cloud downsampling) \
├── lio.py # Main LIO pipeline (LLIO class) \
├── cfg.yml # Configuration file \
└── README.md # This file 

## Prepare the KITTI dataset
data/ \
└── kitti/ \
    └── <sequence_name>/ \
        ├── oxts/ \
        ├── velodyne/ \
        └── calib/

Update cfg.yml accordingly.

## 📊 Output
Estimated trajectory and ground truth.

Optional covariance and registration visualization.

Logs of pose updates and state corrections.