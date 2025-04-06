from transform_lidar_camera import LiDAR2Camera
from detect_obstacles import YoloOD
import glob
import open3d as o3d
import numpy as np
import cv2
import os

def final_pipeline(lidar2cam_obj, yolo_obj, image, point_cloud):
    img = image.copy()
    lidar_img = lidar2cam_obj.show_lidar_on_image(point_cloud[:, :3], image)
    result, pred_bboxes = yolo_obj.run_obstacle_detection(img)
    img_final, _ = lidar2cam_obj.lidar_camera_fusion(pred_bboxes, result)
    return img_final


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
    out = cv2.VideoWriter('out_4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 0.5, (width, height))

    # Create a directory to save detected images
    output_image_dir = "output_images"
    os.makedirs(output_image_dir, exist_ok=True)

    for idx, img_path in enumerate(video_images):
        print(f"Processing frame {idx + 1}/{len(video_images)}")

        # Load calibration for the current frame
        lidar2cam_video = LiDAR2Camera(calib_files[idx])

        # Read image and point cloud
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)

        # Process the frame
        result_frame = final_pipeline(lidar2cam_obj=lidar2cam_video,
                                      yolo_obj=yolo_obj,
                                      image=image,
                                      point_cloud=point_cloud)

        # Convert frame back to BGR for saving
        result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

        # Save the processed image
        output_image_path = os.path.join(output_image_dir, f"{idx:06d}.png")
        cv2.imwrite(output_image_path, result_frame_bgr)

        # Write frame to the video
        out.write(result_frame_bgr)

    # Release the video writer
    out.release()
    print("Video generation and image saving completed successfully.")