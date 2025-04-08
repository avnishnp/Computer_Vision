from transform_lidar_camera import LiDAR2Camera
from detect_obstacles import YoloOD
from bev_visualizer import create_enhanced_bev
import glob
import open3d as o3d
import numpy as np
import cv2
import os
import argparse

def process_single_frame(lidar2cam_obj, yolo_obj, image, point_cloud, save_path=None):
    """
    Process a single frame with full LiDAR-camera fusion and enhanced BEV visualization
    
    Args:
        lidar2cam_obj: LiDAR2Camera object
        yolo_obj: YoloOD object
        image: Input RGB image
        point_cloud: LiDAR point cloud
        save_path: Optional path to save the output
        
    Returns:
        Tuple of (camera_view, bev_view, combined_view)
    """
    # Make a copy of the original image
    img_copy = image.copy()
    
    # Project LiDAR points to image without drawing
    lidar2cam_obj.project_lidar_points_to_image(point_cloud[:, :3], img_copy)
    
    # Run YOLO detection and get bounding boxes
    result_img, pred_bboxes = yolo_obj.run_obstacle_detection(img_copy.copy())
    print(f"Detected {len(pred_bboxes)} objects")
    
    # Perform LiDAR-camera fusion with detected bounding boxes
    camera_view, distances = lidar2cam_obj.lidar_camera_fusion(pred_bboxes, result_img)
    
    # Create enhanced BEV visualization with all bounding boxes
    # Note: The enhanced visualization handles cases where distances are missing
    bev_view = create_enhanced_bev(
        point_cloud=point_cloud, 
        detections=pred_bboxes, 
        distances=distances, 
        x_range=(-30, 30), 
        y_range=(-15, 15)
    )
    
    # Get image dimensions
    height, width, _ = camera_view.shape
    bev_size = (600, 600)  # Size of BEV visualization
    bev_view_resized = cv2.resize(bev_view, bev_size)
    
    # Create combined view
    combined_width = width + bev_size[0]
    combined_height = max(height, bev_size[1])
    combined_view = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Convert RGB to BGR for display
    camera_view_bgr = cv2.cvtColor(camera_view, cv2.COLOR_RGB2BGR)
    
    # Insert images into combined view
    combined_view[:height, :width] = camera_view_bgr
    combined_view[:bev_size[1], width:width+bev_size[0]] = bev_view_resized
    
    # Add object counts by class to the combined view
    class_counts = {}
    for det in pred_bboxes:
        cls_id = int(det[5])
        class_names = {
            0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 
            4: 'Airplane', 5: 'Bus', 6: 'Train', 7: 'Truck',
            8: 'Boat', 9: 'Traffic Light'
        }
        class_name = class_names.get(cls_id, f'Class {cls_id}')
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Add class count summary to combined view
    summary_y = combined_height - 120
    cv2.putText(combined_view, "Detected Objects:", (width + 10, summary_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    summary_y += 30
    for class_name, count in class_counts.items():
        cv2.putText(combined_view, f"{class_name}: {count}", (width + 20, summary_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        summary_y += 25
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, combined_view)
        print(f"Saved combined view to {save_path}")
    
    return camera_view, bev_view, combined_view

def process_dataset(image_dir, point_dir, calib_dir, output_dir="bev_output", 
                   create_video=True, tiny_model=False):
    """
    Process a complete dataset with LiDAR-camera fusion and enhanced BEV visualization
    
    Args:
        image_dir: Directory containing camera images
        point_dir: Directory containing LiDAR point clouds
        calib_dir: Directory containing calibration files
        output_dir: Directory to save output files
        create_video: Whether to create videos from the processed frames
        tiny_model: Whether to use the tiny YOLO model
        
    Returns:
        Paths to the output videos
    """
    # Create glob patterns for files
    image_pattern = os.path.join(image_dir, "*.png")
    point_pattern = os.path.join(point_dir, "*.pcd")
    calib_pattern = os.path.join(calib_dir, "*.txt")
    
    # Get sorted lists of files
    video_images = sorted(glob.glob(image_pattern))
    video_points = sorted(glob.glob(point_pattern))
    calib_files = sorted(glob.glob(calib_pattern))
    
    # Ensure that the number of files matches
    if not (len(video_images) == len(video_points) == len(calib_files)):
        raise ValueError(f"Mismatch in number of files: {len(video_images)} images, "
                        f"{len(video_points)} point clouds, {len(calib_files)} calibration files")
    
    print(f"Found {len(video_images)} frames to process")
    
    # Initialize YOLO object detection
    yolo_obj = YoloOD(tiny_model=tiny_model)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    output_camera_dir = os.path.join(output_dir, "camera")
    output_bev_dir = os.path.join(output_dir, "bev")
    output_combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(output_camera_dir, exist_ok=True)
    os.makedirs(output_bev_dir, exist_ok=True)
    os.makedirs(output_combined_dir, exist_ok=True)
    
    # Get dimensions for video writers
    first_image = cv2.imread(video_images[0])
    height, width, _ = first_image.shape
    bev_size = (600, 600)  # Size of BEV visualization
    combined_width = width + bev_size[0]
    combined_height = max(height, bev_size[1])
    
    # Initialize video writers if needed
    if create_video:
        video_fps = 0.5  # Frames per second
        out_camera = cv2.VideoWriter(os.path.join(output_dir, 'camera_view.mp4'), 
                                    cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (width, height))
        out_bev = cv2.VideoWriter(os.path.join(output_dir, 'bev_view.mp4'), 
                                 cv2.VideoWriter_fourcc(*'mp4v'), video_fps, bev_size)
        out_combined = cv2.VideoWriter(os.path.join(output_dir, 'combined_view.mp4'), 
                                      cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (combined_width, combined_height))
    
    # Process each frame
    for idx, img_path in enumerate(video_images):
        print(f"Processing frame {idx + 1}/{len(video_images)}")
        
        # Load calibration file for this frame
        lidar2cam_video = LiDAR2Camera(calib_files[idx])
        
        # Read the image and point cloud for this frame
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)
        
        try:
            # Process the frame
            camera_view, bev_view, combined_view = process_single_frame(
                lidar2cam_obj=lidar2cam_video,
                yolo_obj=yolo_obj,
                image=image,
                point_cloud=point_cloud
            )
            
            # Save images
            camera_view_bgr = cv2.cvtColor(camera_view, cv2.COLOR_RGB2BGR)
            bev_view_resized = cv2.resize(bev_view, bev_size)
            
            output_camera_path = os.path.join(output_camera_dir, f"{idx:06d}.png")
            output_bev_path = os.path.join(output_bev_dir, f"{idx:06d}.png")
            output_combined_path = os.path.join(output_combined_dir, f"{idx:06d}.png")
            
            cv2.imwrite(output_camera_path, camera_view_bgr)
            cv2.imwrite(output_bev_path, bev_view_resized)
            cv2.imwrite(output_combined_path, combined_view)
            
            # Add to videos if needed
            if create_video:
                out_camera.write(camera_view_bgr)
                out_bev.write(bev_view_resized)
                out_combined.write(combined_view)
                
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Release video writers if needed
    if create_video:
        out_camera.release()
        out_bev.release()
        out_combined.release()
        print("Video generation completed successfully")
    
    print(f"All frames processed. Results saved to {output_dir}")
    
    # Return paths to output videos
    if create_video:
        return {
            "camera": os.path.join(output_dir, 'camera_view.mp4'),
            "bev": os.path.join(output_dir, 'bev_view.mp4'),
            "combined": os.path.join(output_dir, 'combined_view.mp4')
        }
    else:
        return {
            "camera_dir": output_camera_dir,
            "bev_dir": output_bev_dir,
            "combined_dir": output_combined_dir
        }

def main():
    """Command-line interface for processing LiDAR and camera data"""
    parser = argparse.ArgumentParser(description="LiDAR-Camera Fusion with Enhanced BEV Visualization")
    
    parser.add_argument("--image_dir", type=str, default="Early_Sensor_Fusion/data/img", 
                       help="Directory containing camera images")
    parser.add_argument("--point_dir", type=str, default="Early_Sensor_Fusion/data/velodyne", 
                       help="Directory containing LiDAR point clouds")
    parser.add_argument("--calib_dir", type=str, default="Early_Sensor_Fusion/data/calib", 
                       help="Directory containing calibration files")
    parser.add_argument("--output_dir", type=str, default="Early_Sensor_Fusion/output_bev", 
                       help="Directory to save output files")
    parser.add_argument("--no_video", action="store_true", 
                       help="Don't create videos, only save individual frames")
    parser.add_argument("--tiny_model", action="store_true", 
                       help="Use tiny YOLO model for faster processing")
    
    args = parser.parse_args()
    
    # Process the dataset
    process_dataset(
        image_dir=args.image_dir,
        point_dir=args.point_dir,
        calib_dir=args.calib_dir,
        output_dir=args.output_dir,
        create_video=not args.no_video,
        tiny_model=args.tiny_model
    )

if __name__ == "__main__":
    # Check if this script is being run directly
    print("LiDAR-Camera Fusion with Enhanced BEV Visualization")
    print("--------------------------------------------------")
    print("This script provides an improved visualization of LiDAR and camera fusion.")
    print("Bounding boxes for all detected objects will be shown, even if they lack LiDAR points.")
    print("\nTo run the full processing pipeline, use the following command:")
    print("\npython main.py --image_dir data/img --point_dir data/velodyne --calib_dir data/calib\n")
    
    # Run a demo if data directories exist
    image_dir = "data/img"
    point_dir = "data/velodyne"
    calib_dir = "data/calib"
    
    if os.path.exists(image_dir) and os.path.exists(point_dir) and os.path.exists(calib_dir):
        print("Data directories found. Running a demo with the first frame...")
        
        # Get first frame files
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        point_files = sorted(glob.glob(os.path.join(point_dir, "*.pcd")))
        calib_files = sorted(glob.glob(os.path.join(calib_dir, "*.txt")))
        
        if image_files and point_files and calib_files:
            # Initialize objects
            lidar2cam = LiDAR2Camera(calib_files[0])
            yolo_obj = YoloOD(tiny_model=False)
            
            # Load data
            image = cv2.cvtColor(cv2.imread(image_files[0]), cv2.COLOR_BGR2RGB)
            point_cloud = np.asarray(o3d.io.read_point_cloud(point_files[0]).points)
            
            # Process single frame
            os.makedirs("demo_output", exist_ok=True)
            _, _, combined_view = process_single_frame(
                lidar2cam_obj=lidar2cam,
                yolo_obj=yolo_obj,
                image=image,
                point_cloud=point_cloud,
                save_path="demo_output/demo_result.png"
            )
            
            print("\nDemo completed! Result saved to 'demo_output/demo_result.png'")
            print("To see the result, open the saved image file.")
            
            # Show image if we have a display
            try:
                cv2.imshow("LiDAR-Camera Fusion with Enhanced BEV", combined_view)
                print("Press any key to close the window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("Could not display the result in a window (no display available).")
        else:
            print("No image, point cloud, or calibration files found in the data directories.")
    else:
        print("Data directories not found. Please check the paths or run with proper arguments.")
    
    main()