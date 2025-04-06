import numpy as np
import cv2
import matplotlib.pyplot as plt
import statistics
import random

class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)
        P = calibs["P2"]
        self.P = np.reshape(P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(V2C, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        R0 = calibs["R0_rect"]
        self.R0 = np.reshape(R0, [3, 3])

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def cart2hom(self, pts_3d):
        """
        Cartesian to Homogeneous Coordinates
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_velo_to_image(self, pts_3d_velo):
        '''
        Project from Velodyne frame to Camera Frame
        '''
        R0_homo = np.vstack([self.R0, [0, 0, 0]])
        R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])
        p_r0 = np.dot(self.P, R0_homo_2)  # PxR0
        p_r0_rt = np.dot(p_r0, np.vstack((self.V2C, [0, 0, 0, 1])))  # PxROxRT
        pts_3d_homo = np.column_stack([pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))])
        p_r0_rt_x = np.dot(p_r0_rt, np.transpose(pts_3d_homo))  # PxROxRTxX
        pts_2d = np.transpose(p_r0_rt_x)

        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def get_lidar_in_image_fov(self, pc_velo, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
        """ Filter lidar points, keep those in image FOV """
        pts_2d = self.project_velo_to_image(pc_velo)
        fov_inds = (
                (pts_2d[:, 0] < xmax)
                & (pts_2d[:, 0] >= xmin)
                & (pts_2d[:, 1] < ymax)
                & (pts_2d[:, 1] >= ymin)
        )
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
        imgfov_pc_velo = pc_velo[fov_inds, :]
        if return_more:
            return imgfov_pc_velo, pts_2d, fov_inds
        else:
            return imgfov_pc_velo

    def project_lidar_points_to_image(self, pc_velo, img):
        """
        Project LiDAR points to image WITHOUT drawing them
        This function only computes the projection and saves the data for later use
        """
        imgfov_pc_velo, pts_2d, fov_inds = self.get_lidar_in_image_fov(
            pc_velo, 0, 0, img.shape[1], img.shape[0], True)
        
        # Store the projected points and their corresponding 3D coordinates
        self.imgfov_pc_velo = imgfov_pc_velo  # 3D points in LiDAR frame
        self.imgfov_pts_2d = pts_2d[fov_inds, :]  # 2D points in image frame
        
        return img  # Return the image without any modifications

    def filter_outliers(self, distances):
        """Filter outliers using mean and standard deviation"""
        if len(distances) < 2:
            return distances
            
        inliers = []
        mu = statistics.mean(distances)
        std = statistics.stdev(distances)
        for x in distances:
            if abs(x - mu) < std:
                # This is an INLIER
                inliers.append(x)
        return inliers if inliers else distances

    def get_best_distance(self, distances, technique="closest"):
        """Get the best representative distance based on the specified technique"""
        if not distances:
            return None
            
        if technique == "closest":
            return min(distances)
        elif technique == "average":
            return statistics.mean(distances)
        elif technique == "random":
            return random.choice(distances)
        else:
            return statistics.median(sorted(distances))

    def lidar_camera_fusion(self, pred_bboxes, image):
        """
        Performs LiDAR-camera fusion by associating LiDAR points with YOLO-detected bounding boxes.
        Only draws LiDAR points that are within bounding boxes.
        Improves text positioning to avoid overlapping.
        """
        img_bis = image.copy()
        h, w, _ = img_bis.shape

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        all_distances = []

        print(f"Processing {len(pred_bboxes)} bounding boxes")
        
        # First draw the bounding boxes without any points
        for box in pred_bboxes:
            if len(box) < 6:
                print(f"Skipping invalid box format: {box}")
                continue

            x1, y1, x2, y2, conf, cls = box
            
            # Ensure bounding box coordinates are integers and within valid range
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w - 1, int(x2))
            y2 = min(h - 1, int(y2))

            if x2 <= x1 or y2 <= y1:
                print(f"Invalid box coordinates after adjustment: {x1}, {y1}, {x2}, {y2}")
                continue

            # Draw bounding box
            cv2.rectangle(img_bis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Check if we have LiDAR points projected
        if not hasattr(self, 'imgfov_pts_2d') or not hasattr(self, 'imgfov_pc_velo'):
            print("No LiDAR points projected. Run project_lidar_points_to_image first.")
            return img_bis, []
        
        # Now process each bounding box and draw points only within boxes
        for box in pred_bboxes:
            if len(box) < 6:
                continue

            x1, y1, x2, y2, conf, cls = box
            
            # Ensure bounding box coordinates are integers and within valid range
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w - 1, int(x2))
            y2 = min(h - 1, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            print(f"Processing box: ({x1}, {y1}, {x2}, {y2}), confidence: {conf:.2f}, class: {int(cls)}")
            box_distances = []

            # Draw points only if they're inside this bounding box
            for i in range(self.imgfov_pts_2d.shape[0]):
                point_x, point_y = self.imgfov_pts_2d[i]
                if x1 <= point_x <= x2 and y1 <= point_y <= y2:
                    depth = self.imgfov_pc_velo[i, 0]
                    box_distances.append(depth)
                    color_idx = min(int(510.0 / depth), 255)
                    color = tuple(map(int, cmap[color_idx]))

                    cv2.circle(
                        img_bis,
                        (int(np.round(point_x)), int(np.round(point_y))),
                        2,
                        color=color,
                        thickness=-1,
                    )

            if len(box_distances) > 0:
                filtered_distances = self.filter_outliers(box_distances)
                if filtered_distances:
                    best_distance = self.get_best_distance(filtered_distances, technique="average")
                    all_distances.append(best_distance)

                    # Add distance label with improved positioning
                    # Position the text at the bottom of the bounding box instead of the top
                    # This helps avoid overlapping with detection labels
                    label = f"{best_distance:.2f}m"
                    
                    # Calculate text size to better position it
                    text_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )[0]
                    
                    # Position text at the bottom of the box
                    text_x = x1 + 5
                    text_y = y2 + text_size[1] + 5  # Position below the box
                    
                    # Check if text would go off screen
                    if text_y >= h:
                        # If it would go off screen, position it inside the bottom of the box
                        text_y = y2 - 5
                    
                    # Add a small background rectangle for better visibility
                    cv2.rectangle(
                        img_bis,
                        (text_x - 2, text_y - text_size[1] - 2),
                        (text_x + text_size[0] + 2, text_y + 2),
                        (0, 0, 0),  # Black background
                        -1  # Filled rectangle
                    )
                    
                    # Draw the text
                    cv2.putText(
                        img_bis,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,  # Slightly smaller font
                        (255, 255, 255),  # White text
                        2,
                        cv2.LINE_AA,
                    )
            else:
                # No points found in this box
                print(f"No LiDAR points found in box {x1}, {y1}, {x2}, {y2}")
            
        return img_bis, all_distances