import os
import cv2
import argparse
import numpy as np
import pyntcloud as pc
import pandas as pd

from scipy.interpolate import RectBivariateSpline
from feature import BuildFeatureTrack
from camera_pose import EstimateCameraPose
from camera_pose import Triangulation
from camera_pose import EvaluateCheirality
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='im')
    args = parser.parse_args()

    intrinsic_parameters = np.asarray([
        [3228.13099, 0, 600],
        [0, 3238.72283, 450],
        [0, 0, 1]
    ])

    List_Img = os.listdir(args.img_dir)
    List_Img.sort()
    Num_Images = len(List_Img)
    img_shape = cv2.imread(os.path.join(args.img_dir, List_Img[0])).shape
    height = img_shape[0]
    width = img_shape[1]

    Images = np.empty((Num_Images, height, width, 3), dtype=np.uint8)
    for i in range(Num_Images):
        im = cv2.imread(os.path.join(args.img_dir, List_Img[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Images[i, :, :, :] = im

    track = BuildFeatureTrack(Images, intrinsic_parameters)

    track_first = track[0, :, :]
    track_second = track[1, :, :]

    R, C, X = EstimateCameraPose(track_first, track_second)

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    Poses = np.zeros((Num_Images, 3, 4))

    Poses[0] = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0]])
    Poses[1] = np.hstack([R, -(R @ C).reshape((3, 1))])

    ransac_iteration = 1000
    ransac_threshold = 0.5
    for i in range(2, Num_Images):
        X_mask = np.logical_and(np.logical_and(X[:, 0] != -1, X[:, 1] != -1), X[:, 2] != -1)

        track_i = track[i, :, :]
        track_mask_i = np.logical_and(track_i[:, 0] != -1, track_i[:, 1] != -1)

        mask = np.logical_and(X_mask, track_mask_i)
        R, C, inlier = PnP_RANSAC(X[mask], track_i[mask], ransac_iteration, ransac_threshold)
        R, C = PnP_nl(R, C, X[mask], track_i[mask])

        Poses[i] = np.hstack([R, -(R @ C).reshape((3, 1))])

        for j in range(i):
            track_j = track[j, :, :]
            track_mask_j = np.logical_and(track_j[:, 0] != -1, track_j[:, 1] != -1)

            mask = np.logical_and(np.logical_and(track_mask_i, track_mask_j), ~X_mask)
            mask_pos = np.asarray(np.nonzero(mask)[0])

            print('Triangulation for image %d and %d' % (i, j))
            missing_X = Triangulation(Poses[i], Poses[j], track_i[mask], track_j[mask])
            missing_X = Triangulation_nl(missing_X, Poses[i], Poses[j], track_i[mask], track_j[mask])

            valid_pos = EvaluateCheirality(Poses[i], Poses[j], missing_X)
            X[mask_pos[valid_pos]] = missing_X[valid_pos]

        valid_ps = X[:, 0] != -1
        X_current = X[valid_ps, :]
        track_current = track[:i + 1, valid_ps, :]
        P_latest, X_latest = RunBundleAdjustment(Poses[:i + 1, :, :], X_current, track_current)
        Poses[:i + 1, :, :] = P_latest
        X[valid_ps, :] = X_latest

    # Save camera poses as PLY file
    camera_poses = np.zeros((Num_Images, 3))
    for i in range(Num_Images):
        R = Poses[i, :, :3]
        C = -R.T @ Poses[i, :, 3]
        camera_poses[i] = C

    camera_df = pd.DataFrame(camera_poses, columns=["x", "y", "z"])
    camera_cloud = pc.PyntCloud(camera_df)
    camera_cloud.to_file(os.path.join(output_dir, "camera_poses.ply"))

    # Save point cloud as PLY file
    valid_points = np.sqrt(np.sum(X_latest ** 2, axis=1)) < 200
    point_cloud_df = pd.DataFrame(X_latest[valid_points], columns=["x", "y", "z"])
    point_cloud = pc.PyntCloud(point_cloud_df)
    point_cloud.to_file(os.path.join(output_dir, "point_cloud.ply"))
