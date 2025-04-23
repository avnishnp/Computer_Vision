# Purpose of this file : Loop over all frames in a Waymo Open Dataset file,
#                        detect and track objects, perform 3D scene reconstruction
#                        and visualize results

##################
## Imports

## general package imports
import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
from tqdm import tqdm

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2

## 3d object detection
import tools.pcl as pcl
import tools.lidar_detect as det
import eval

import tools.misc as tools 
from tools.helpers import save_object_to_file, load_object_from_file, make_exec_list

from kalman_filter.filter import Filter
from kalman_filter.track_management import Trackmanagement
from kalman_filter.association import Association
from kalman_filter.measurements import Sensor, Measurement
from plots import plot_tracks, plot_rmse, make_movie
import kalman_filter.params as params 

##################
## Scene Reconstruction Implementation

class SceneReconstruction:
    def __init__(self, voxel_size=0.2):
        """
        Initialize the 3D scene reconstruction
        
        :param voxel_size: Size of voxels for downsampling the point cloud
        """
        self.voxel_size = voxel_size
        self.global_map = o3d.geometry.PointCloud()
        self.poses = []  # Store all poses
        self.frame_count = 0
        self.first_pose = None  # Store first pose for reference
        
        # For visualization
        self.vis = None
        self.is_visualization_initialized = False
    
    def preprocess_point_cloud(self, point_cloud, voxel_size):
        """
        Downsample point cloud and estimate normals
        
        :param point_cloud: Input point cloud
        :param voxel_size: Size of voxels for downsampling
        :return: Processed point cloud
        """
        print(":: Downsample with voxel size {:.3f}.".format(voxel_size))
        pcd_down = point_cloud.voxel_down_sample(voxel_size)
        
        # Estimate normals
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        return pcd_down
    
    def prepare_point_cloud(self, lidar_pcl):
        """
        Convert numpy point cloud to Open3D point cloud
        
        :param lidar_pcl: Numpy array of point cloud [x, y, z, intensity]
        :return: Open3D point cloud
        """
        # Extract points (ignore intensity for reconstruction)
        points = lidar_pcl[:, 0:3]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Random colors based on XYZ coordinates for visualization
        colors = np.zeros_like(points)
        colors[:, 0] = 0.5 + points[:, 0] / 100.0
        colors[:, 1] = 0.5 + points[:, 1] / 100.0
        colors[:, 2] = 0.5 + points[:, 2] / 100.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Preprocess
        return self.preprocess_point_cloud(pcd, self.voxel_size)
        
    def update_map(self, lidar_pcl, pose):
        """
        Update the global map with new lidar scan using the provided pose
        
        :param lidar_pcl: New lidar point cloud (numpy array)
        :param pose: 4x4 transformation matrix representing vehicle pose
        """
        # Store the pose
        if self.first_pose is None:
            self.first_pose = pose.copy()
            # First frame is the origin of our map
            relative_pose = np.eye(4)
        else:
            # Calculate pose relative to the first frame
            relative_pose = np.linalg.inv(self.first_pose) @ pose
        
        self.poses.append(relative_pose.copy())
        
        # Convert to Open3D point cloud
        current_pcd = self.prepare_point_cloud(lidar_pcl)
        
        # Transform current point cloud to global frame
        current_pcd_global = copy.deepcopy(current_pcd)
        current_pcd_global.transform(relative_pose)
        
        # Add to global map
        self.global_map += current_pcd_global
        
        # Down-sample global map to manage size
        if self.frame_count % 10 == 0:
            self.global_map = self.global_map.voxel_down_sample(self.voxel_size * 2)
        
        self.frame_count += 1
    
    def init_visualization(self):
        """Initialize Open3D visualization"""
        if not self.is_visualization_initialized:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="3D Scene Reconstruction", width=1280, height=720)
            self.is_visualization_initialized = True
            
            # Add global map to visualization
            self.vis.add_geometry(self.global_map)
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            self.vis.add_geometry(coord_frame)
    
    # def update_visualization(self):
    #     """Update visualization"""
    #     if not self.is_visualization_initialized:
    #         self.init_visualization()
        
    #     self.vis.remove_geometry(self.global_map, reset_bounding_box=False)
    #     self.vis.add_geometry(self.global_map)
    #     self.vis.poll_events()
    #     self.vis.update_renderer()
    
    def update_visualization(self):
        """Update Open3D visualization with point cloud, ego-car frame, and trajectory"""
        if not self.is_visualization_initialized:
            self.init_visualization()

        # Clear existing geometries (point cloud will be re-added)
        self.vis.clear_geometries()

        # Add current global point cloud
        self.vis.add_geometry(self.global_map)

        # ───────────────────────────────
        # 🧭 Add car's current coordinate frame
        if self.poses:
            car_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
            car_frame.transform(self.poses[-1])  # Use latest pose
            self.vis.add_geometry(car_frame)

        # ───────────────────────────────
        # 📈 Add trajectory line
        if len(self.poses) > 1:
            pts = [pose[:3, 3] for pose in self.poses]
            lines = [[i, i + 1] for i in range(len(pts) - 1)]
            colors = [[0, 0, 1] for _ in lines]  # Blue line

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(pts)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            self.vis.add_geometry(line_set)

        # ───────────────────────────────
        # Refresh visualization
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.05)  # Add delay to slow down rendering and make it visible



    
    def save_reconstruction(self, output_path="reconstruction.pcd"):
        """
        Save the reconstructed point cloud in either .pcd or .ply format

        :param output_path: Path to save the point cloud (must end with .pcd or .ply)
        """
        ext = os.path.splitext(output_path)[1].lower()
        if ext not in [".pcd", ".ply"]:
            raise ValueError("Output file must be .pcd or .ply")

        print(f"Saving reconstruction to {output_path}")
        o3d.io.write_point_cloud(output_path, self.global_map)

    
    def visualize_trajectory(self, output_path="trajectory.png"):
        """
        Visualize trajectory as a line graph
        
        :param output_path: Path to save the visualization
        """
        # Extract translation components from poses
        translations = np.array([pose[:3, 3] for pose in self.poses])
        
        # Plot the trajectory
        plt.figure(figsize=(10, 8))
        plt.plot(translations[:, 0], translations[:, 1], 'b-', linewidth=2)
        plt.plot(translations[0, 0], translations[0, 1], 'ro', markersize=10, label='Start')
        plt.plot(translations[-1, 0], translations[-1, 1], 'go', markersize=10, label='End')
        plt.grid(True)
        plt.axis('equal')
        plt.title('Vehicle Trajectory')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        print(f"Trajectory visualization saved to {output_path}")
    
    def close_visualization(self):
        """Close visualization"""
        if self.is_visualization_initialized:
            self.vis.destroy_window()
            self.is_visualization_initialized = False

##################
## Set parameters and perform initializations

## Select Waymo Open Dataset file and frame numbers
data_filename = 'individual_files_training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
#data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
# data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord' # Sequence 3
show_only_frames = [0, 200] # show only frames in interval for debugging

## Prepare Waymo Open Dataset file for loading
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)  # initialize dataset iterator

## Initialize object detection
configs_det = det.load_configs(model_name='fpn_resnet') # options are 'darknet', 'fpn_resnet'
model_det = det.create_model(configs_det)

configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

## Uncomment this setting to restrict the y-range in the final project
configs_det.lim_y = [-25, 25]

## Initialize tracking
KF = Filter() # set up Kalman filter 
association = Association() # init data association
manager = Trackmanagement() # init track manager
lidar = None # init lidar sensor object
camera = None # init camera sensor object
np.random.seed(10) # make random values predictable

## Initialize scene reconstruction (optional)
enable_reconstruction = True  # Set to False to disable reconstruction
if enable_reconstruction:
    scene_reconstruction = SceneReconstruction(voxel_size=0.1)


## Selective execution and visualization
exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'] # options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
exec_tracking = [] #['perform_tracking'] # options are 'perform_tracking' keep this empty for 3d reconstruction map
# exec_tracking= ['perform_tracking'] # options are 'perform_tracking' keep this for kalman filter track
exec_visualization = [] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
# exec_visualization = ['show_tracks']
exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)


##################
## Perform detection & tracking over all selected frames

cnt_frame = 0 
all_labels = []
det_performance_all = [] 
if 'show_tracks' in exec_list:    
    fig, (ax2, ax) = plt.subplots(1,2) # init track plot
    
while True:
    try:
        ## Get next frame from Waymo dataset
        frame = next(datafile_iter)
        if cnt_frame < show_only_frames[0]:
            cnt_frame = cnt_frame + 1
            continue
        elif cnt_frame > show_only_frames[1]:
            print('reached end of selected frames')
            break
        
        print('------------------------------')
        print('processing frame #' + str(cnt_frame))

        #################################
        ## Perform 3D object detection

        ## Extract calibration data and front camera image from frame
        lidar_name = dataset_pb2.LaserName.TOP
        camera_name = dataset_pb2.CameraName.FRONT
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)        
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)
        if 'load_image' in exec_list:
            image = tools.extract_front_camera_image(frame) 

        ## Compute lidar point-cloud from range image    
        if 'pcl_from_rangeimage' in exec_list:
            print('computing point-cloud from lidar range image')
            lidar_pcl = tools.pcl_from_range_image(frame, lidar_name)
        else:
            print('loading lidar point-cloud from result file')
            lidar_pcl = load_object_from_file(results_fullpath, data_filename, 'lidar_pcl', cnt_frame)
            
        ## Add point cloud to scene reconstruction
        if enable_reconstruction:
            # Update the 3D reconstruction with the current lidar point cloud
            pose = np.array(frame.pose.transform).reshape(4, 4)
            scene_reconstruction.update_map(lidar_pcl, pose)
            
            # Update visualization every 5 frames to avoid slowing down processing
            # if cnt_frame % 5 == 0:
            scene_reconstruction.update_visualization()
            
        ## Compute lidar birds-eye view (bev)
        if 'bev_from_pcl' in exec_list:
            print('computing birds-eye view from lidar pointcloud')
            lidar_bev = pcl.bev_from_pcl(lidar_pcl, configs_det)
        else:
            print('loading birds-eve view from result file')
            lidar_bev = load_object_from_file(results_fullpath, data_filename, 'lidar_bev', cnt_frame)

        ## 3D object detection
        if (configs_det.use_labels_as_objects==True):
            print('using groundtruth labels as objects')
            detections = tools.convert_labels_into_objects(frame.laser_labels, configs_det)
        else:
            if 'detect_objects' in exec_list:
                print('detecting objects in lidar pointcloud')   
                detections = det.detect_objects(lidar_bev, model_det, configs_det)
            else:
                print('loading detected objects from result file')
                # load different data for final project vs. mid-term project
                if 'perform_tracking' in exec_list:
                    detections = load_object_from_file(results_fullpath, data_filename, 'detections', cnt_frame)
                else:
                    detections = load_object_from_file(results_fullpath, data_filename, 'detections_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame)

        ## Validate object labels
        if 'validate_object_labels' in exec_list:
            print("validating object labels")
            valid_label_flags = tools.validate_object_labels(frame.laser_labels, lidar_pcl, configs_det, 0 if configs_det.use_labels_as_objects==True else 10)
        else:
            print('loading object labels and validation from result file')
            valid_label_flags = load_object_from_file(results_fullpath, data_filename, 'valid_labels', cnt_frame)            

        ## Performance evaluation for object detection
        if 'measure_detection_performance' in exec_list:
            print('measuring detection performance')
            det_performance = eval.measure_detection_performance(detections, frame.laser_labels, valid_label_flags, configs_det.min_iou)     
        else:
            print('loading detection performance measures from file')
            # load different data for final project vs. mid-term project
            if 'perform_tracking' in exec_list:
                det_performance = load_object_from_file(results_fullpath, data_filename, 'det_performance', cnt_frame)
            else:
                det_performance = load_object_from_file(results_fullpath, data_filename, 'det_performance_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame)  

        det_performance_all.append(det_performance) # store all evaluation results in a list for performance assessment at the end
        

        ## Visualization for object detection
        if 'show_range_image' in exec_list:
            img_range = pcl.show_range_image(frame, lidar_name)
            img_range = img_range.astype(np.uint8)
            cv2.imshow('range_image', img_range)
            cv2.waitKey(vis_pause_time)

        if 'show_pcl' in exec_list:
            pcl.show_pcl(lidar_pcl)

        if 'show_bev' in exec_list:
            tools.show_bev(lidar_bev, configs_det)  
            cv2.waitKey(vis_pause_time)          

        if 'show_labels_in_image' in exec_list:
            img_labels = tools.project_labels_into_camera(camera_calibration, image, frame.laser_labels, valid_label_flags, 0.5)
            cv2.imshow('img_labels', img_labels)
            cv2.waitKey(vis_pause_time)

        if 'show_objects_and_labels_in_bev' in exec_list:
            tools.show_objects_labels_in_bev(detections, frame.laser_labels, lidar_bev, configs_det)
            cv2.waitKey(vis_pause_time)         

        if 'show_objects_in_bev_labels_in_camera' in exec_list:
            tools.show_objects_in_bev_labels_in_camera(detections, lidar_bev, image, frame.laser_labels, valid_label_flags, camera_calibration, configs_det)
            cv2.waitKey(vis_pause_time)               


        #################################
        ## Perform tracking
        if 'perform_tracking' in exec_list:
            # set up sensor objects
            if lidar is None:
                lidar = Sensor('lidar', lidar_calibration)
            if camera is None:
                camera = Sensor('camera', camera_calibration)
            
            # preprocess lidar detections
            meas_list_lidar = []
            for detection in detections:
                # check if measurement lies inside specified range
                if detection[1] > configs_det.lim_x[0] and detection[1] < configs_det.lim_x[1] and detection[2] > configs_det.lim_y[0] and detection[2] < configs_det.lim_y[1]:
                    meas_list_lidar = lidar.generate_measurement(cnt_frame, detection[1:], meas_list_lidar)
                
            # preprocess camera detections
            meas_list_cam = []
            for label in frame.camera_labels[0].labels:
                if(label.type == label_pb2.Label.Type.TYPE_VEHICLE):
                
                    box = label.box
                    # use camera labels as measurements and add some random noise
                    z = [box.center_x, box.center_y, box.width, box.length]
                    z[0] = z[0] + np.random.normal(0, params.sigma_cam_i) 
                    z[1] = z[1] + np.random.normal(0, params.sigma_cam_j)
                    meas_list_cam = camera.generate_measurement(cnt_frame, z, meas_list_cam)
            
            # Kalman prediction
            for track in manager.track_list:
                print('predict track', track.id)
                KF.predict(track)
                track.set_t((cnt_frame - 1)*0.1) # save next timestamp
                
            # associate all lidar measurements to all tracks
            association.associate_and_update(manager, meas_list_lidar, KF)
            
            # associate all camera measurements to all tracks
            association.associate_and_update(manager, meas_list_cam, KF)
            
            # save results for evaluation
            result_dict = {}
            for track in manager.track_list:
                result_dict[track.id] = track
            manager.result_list.append(copy.deepcopy(result_dict))
            label_list = [frame.laser_labels, valid_label_flags]
            all_labels.append(label_list)
            
            # visualization
            if 'show_tracks' in exec_list:
                fig, ax, ax2 = plot_tracks(fig, ax, ax2, manager.track_list, meas_list_lidar, frame.laser_labels, 
                                        valid_label_flags, image, camera, configs_det)
                if 'make_tracking_movie' in exec_list:
                    # save track plots to file
                    fname = results_fullpath + '/tracking%03d.png' % cnt_frame
                    print('Saving frame', fname)
                    fig.savefig(fname)

        # increment frame counter
        cnt_frame = cnt_frame + 1    

    except StopIteration:
        # if StopIteration is raised, break from loop
        print("StopIteration has been raised\n")
        break

#################################
## Finalize reconstruction

if enable_reconstruction:
    # Save final reconstruction
    output_pcd_path = os.path.join(results_fullpath, "reconstruction.pcd")
    output_ply_path = os.path.join(results_fullpath, "reconstruction.ply")

    scene_reconstruction.save_reconstruction(output_pcd_path)
    scene_reconstruction.save_reconstruction(output_ply_path)
    
    # Visualize trajectory
    scene_reconstruction.visualize_trajectory(os.path.join(results_fullpath, "trajectory.png"))
    
    # Show final reconstruction
    print("Showing final reconstruction. Close the window to continue.")
    scene_reconstruction.update_visualization()
    time.sleep(5)  # Give time to view the final result
    scene_reconstruction.close_visualization()


#################################
## Post-processing

## Evaluate object detection performance
if 'show_detection_performance' in exec_list:
    eval.compute_performance_stats(det_performance_all)

## Plot RMSE for all tracks
if 'show_tracks' in exec_list:
    plot_rmse(manager, all_labels, configs_det)

## Make movie from tracking results    
if 'make_tracking_movie' in exec_list:
    make_movie(results_fullpath)