import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from  transform_lidar_camera import LiDAR2Camera
from detect_obstacles import YoloOD
from ipm import CameraToBEV
import glob
import open3d as o3d
import numpy as np
import cv2

def final_pipeline(lidar2cam_obj, cam2bev_obj, yolo_obj, image, point_cloud):
    """
    Performs LiDAR-camera fusion with object detection and BEV transformation.
    
    Args:
        lidar2cam_obj: LiDAR2Camera object
        cam2bev_obj: CameraToBEV object
        yolo_obj: YoloOD object
        image: Input RGB image
        point_cloud: LiDAR point cloud
        
    Returns:
        Tuple of (camera_view with fusion, BEV image, combined view)
    """
    # Make a copy of the original image for processing
    img = image.copy()

    # Run YOLO detection and get bounding boxes
    result, pred_bboxes = yolo_obj.run_obstacle_detection(img)
    print(f"Detected {len(pred_bboxes)} objects")

    # Project LiDAR points to image without drawing them
    lidar2cam_obj.project_lidar_points_to_image(point_cloud[:, :3], img)

    # Perform LiDAR-camera fusion with detected bounding boxes
    camera_view, distances = lidar2cam_obj.lidar_camera_fusion(pred_bboxes, result)
    
    # Create BEV using Inverse Perspective Mapping
    bev_image = cam2bev_obj.image_to_bev(image)
    
    # Add grid to BEV for better spatial context
    bev_with_grid = cam2bev_obj.add_grid_to_bev(bev_image)
    
    # Project detected objects to BEV
    class_names = {
        0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 
        4: 'Airplane', 5: 'Bus', 6: 'Train', 7: 'Truck',
        8: 'Boat', 9: 'Traffic Light'
    }
    bev_with_objects = cam2bev_obj.project_bboxes_to_bev(
        bev_with_grid, pred_bboxes, distances, class_names
    )
    
    # Create combined view with camera and BEV side by side
    h, w = camera_view.shape[:2]
    bev_h, bev_w = bev_with_objects.shape[:2]
    
    # Resize BEV to match camera height
    scale_factor = h / bev_h
    bev_resized = cv2.resize(bev_with_objects, (int(bev_w * scale_factor), h))
    
    # Create the combined image
    combined_w = w + bev_resized.shape[1]
    combined = np.zeros((h, combined_w, 3), dtype=np.uint8)
    
    # Place the images side by side
    combined[:, :w] = camera_view
    combined[:, w:] = bev_resized
    
    # Add separator line
    cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
    
    # Add titles
    cv2.putText(combined, "Camera View with LiDAR Fusion", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Bird's Eye View (IPM)", (w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return camera_view, bev_with_objects, combined


if __name__ == '__main__':
    video_images = sorted(glob.glob("data/img/*.png"))
    video_points = sorted(glob.glob("data/velodyne/*.pcd"))
    calib_files = sorted(glob.glob("data/calib/*.txt"))

    # Ensure that the number of images, point clouds, and calibration files match
    if not (len(video_images) == len(video_points) == len(calib_files)):
        raise ValueError("Mismatch in number of images, point clouds, or calibration files!")

    # Initialize YOLO object detection
    yolo_obj = YoloOD(tiny_model=False)

    # Read the first image to get the frame size
    first_image = cv2.imread(video_images[0])
    height, width, _ = first_image.shape

    # Create output directories
    output_dir = "IPM/ipm_output"
    output_camera_dir = os.path.join(output_dir, "camera")
    output_bev_dir = os.path.join(output_dir, "bev")
    output_combined_dir = os.path.join(output_dir, "combined")
    
    os.makedirs(output_camera_dir, exist_ok=True)
    os.makedirs(output_bev_dir, exist_ok=True)
    os.makedirs(output_combined_dir, exist_ok=True)

    # Initialize VideoWriter for each view
    bev_size = (600, 600)  # Size of BEV image
    combined_width = width + bev_size[0]  # Width of combined view
    
    camera_video = cv2.VideoWriter(os.path.join(output_dir, 'camera_fusion.mp4'), 
                                 cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
    bev_video = cv2.VideoWriter(os.path.join(output_dir, 'bev.mp4'), 
                               cv2.VideoWriter_fourcc(*'mp4v'), 5, bev_size)
    combined_video = cv2.VideoWriter(os.path.join(output_dir, 'combined.mp4'), 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 5, (combined_width, height))

    for idx, img_path in enumerate(video_images):
        print(f"Processing frame {idx + 1}/{len(video_images)}")

        # Load calibration file for this frame
        lidar2cam_obj = LiDAR2Camera(calib_files[idx])
        
        # Initialize Camera to BEV transformer
        cam2bev_obj = CameraToBEV(
            calib_file=calib_files[idx],
            dst_size=bev_size,
            x_range=(-20, 20),
            y_range=(-20, 20)
        )

        # Read the image and point cloud for this frame
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # Visualize the source region
        img_with_region = cam2bev_obj.draw_source_region(image)
        # cv2.imshow("IPM Source Region", cv2.cvtColor(img_with_region, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(10)
        # Convert to BGR for saving
        img_with_region_bgr = cv2.cvtColor(img_with_region, cv2.COLOR_RGB2BGR)

        # Create a folder to save the source region images
        ipm_region_dir = os.path.join(output_dir, "ipm_source_regions")
        os.makedirs(ipm_region_dir, exist_ok=True)

        # Save the image
        region_save_path = os.path.join(ipm_region_dir, f"{idx:06d}.png")
        cv2.imwrite(region_save_path, img_with_region_bgr)
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)

        # Run the final pipeline
        try:
            camera_view, bev_view, combined_view = final_pipeline(
                lidar2cam_obj=lidar2cam_obj,
                cam2bev_obj=cam2bev_obj,
                yolo_obj=yolo_obj,
                image=image,
                point_cloud=point_cloud
            )
            
            # Resize BEV to consistent size
            bev_view_resized = cv2.resize(bev_view, bev_size)
            
            # Convert to BGR for saving with OpenCV
            camera_view_bgr = cv2.cvtColor(camera_view, cv2.COLOR_RGB2BGR)
            
            # Save frames
            output_camera_path = os.path.join(output_camera_dir, f"{idx:06d}.png")
            output_bev_path = os.path.join(output_bev_dir, f"{idx:06d}.png")
            output_combined_path = os.path.join(output_combined_dir, f"{idx:06d}.png")
            
            cv2.imwrite(output_camera_path, camera_view_bgr)
            cv2.imwrite(output_bev_path, bev_view_resized)
            cv2.imwrite(output_combined_path, combined_view)
            
            # Add to videos
            camera_video.write(camera_view_bgr)
            bev_video.write(bev_view_resized)
            combined_video.write(combined_view)
            
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Release video writers
    camera_video.release()
    bev_video.release()
    combined_video.release()
    
    print("Video generation and image saving completed successfully.")
    print(f"Results saved to {output_dir}")