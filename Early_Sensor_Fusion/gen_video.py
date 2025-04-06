from transform_lidar_camera import LiDAR2Camera
from detect_obstacles import YoloOD
import glob
import open3d as o3d
import numpy as np
import cv2
import os

def final_pipeline(lidar2cam_obj, yolo_obj, image, point_cloud):
    """
    Performs LiDAR-camera fusion with object detection.
    Only projects LiDAR points within detected bounding boxes.
    """
    # Make a copy of the original image for processing
    img = image.copy()

    # Run YOLO detection and get bounding boxes
    result, pred_bboxes = yolo_obj.run_obstacle_detection(img)
    print(f"Detected {len(pred_bboxes)} objects")

    # Debug: Print raw YOLO bounding boxes
    for box in pred_bboxes:
        print(f"Raw Bounding Box: {box}")

    # Project LiDAR points to image without drawing them
    # This step is needed to compute the projections for later use
    lidar2cam_obj.project_lidar_points_to_image(point_cloud[:, :3], img)

    # Perform LiDAR-camera fusion with detected bounding boxes
    # This will only draw points within the bounding boxes
    img_final, distances = lidar2cam_obj.lidar_camera_fusion(pred_bboxes, img)

    return img_final, distances


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

    # Initialize VideoWriter
    out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 0.5, (width, height))

    # Create a directory to save detected images
    output_image_dir = "output_images"
    os.makedirs(output_image_dir, exist_ok=True)

    for idx, img_path in enumerate(video_images):
        print(f"Processing frame {idx + 1}/{len(video_images)}")

        # Load calibration file for this frame
        lidar2cam_video = LiDAR2Camera(calib_files[idx])

        # Read the image and point cloud for this frame
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)

        # Run the final pipeline
        try:
            result_frame, distances = final_pipeline(
                lidar2cam_obj=lidar2cam_video,
                yolo_obj=yolo_obj,
                image=image,
                point_cloud=point_cloud,
            )
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            continue

        if not isinstance(result_frame, np.ndarray):
            raise TypeError(f"Frame {idx} returned a non-image type: {type(result_frame)}")

        # Convert to BGR for saving
        result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

        # Save the processed frame as an image
        output_image_path = os.path.join(output_image_dir, f"{idx:06d}.png")
        cv2.imwrite(output_image_path, result_frame_bgr)

        # Write the frame to the video
        out.write(result_frame_bgr)

    # Release the video writer
    out.release()
    print("Video generation and image saving completed successfully.")
