import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class CameraToBEV:
    """
    Class for camera to Bird's Eye View (BEV) transformation using Inverse Perspective Mapping (IPM)
    """
    def __init__(self, calib_file=None, src_points=None, dst_size=(600, 600), 
                 x_range=(-20, 20), y_range=(-20, 20)):
        """
        Initialize the CameraToBEV transformer
        
        Args:
            calib_file: Optional calibration file (KITTI format)
            src_points: Optional source points for perspective transform (if None, will be estimated)
            dst_size: Size of the output BEV image (width, height)
            x_range: Range in meters along x-axis (forward) in BEV
            y_range: Range in meters along y-axis (left/right) in BEV
        """
        self.calib_file = calib_file
        self.src_points = src_points
        self.dst_size = dst_size
        self.x_range = x_range
        self.y_range = y_range
        
        # Resolution in meters per pixel
        self.x_resolution = (x_range[1] - x_range[0]) / dst_size[0]
        self.y_resolution = (y_range[1] - y_range[0]) / dst_size[1]
        
        # Initialize calibration data
        self.P = None
        if calib_file:
            self.load_calibration(calib_file)
        
        # Pre-compute transformations if possible
        self.M = None  # Perspective transform matrix
    
    def load_calibration(self, calib_file):
        """Load calibration from file (KITTI format)"""
        try:
            data = {}
            with open(calib_file, "r") as f:
                for line in f.readlines():
                    line = line.rstrip()
                    if len(line) == 0:
                        continue
                    key, value = line.split(":", 1)
                    try:
                        data[key] = np.array([float(x) for x in value.split()])
                    except ValueError:
                        pass
                        
            # Get projection matrix
            if "P2" in data:
                self.P = np.reshape(data["P2"], [3, 4])
            elif "P2:" in data:
                self.P = np.reshape(data["P2:"], [3, 4])
                
            print(f"Loaded calibration from {calib_file}")
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            print("Using default camera parameters")
    
    def estimate_transform(self, image):
        """
        Estimate the perspective transform based on image size and calibration
        
        Args:
            image: Input image to determine dimensions
        
        Returns:
            Transformation matrix M
        """
        h, w = image.shape[:2]
        
        # If source points are already defined, use them
        if self.src_points is not None:
            src_points = self.src_points
        else:
            # Otherwise, estimate source points based on image dimensions
            # These values are a heuristic and may need adjustment for your specific camera
            vanishing_point_y = int(h * 0.45)  # Horizon line
            
            # Define the source points in the image (trapezoid)
            src_points = np.array([
            [w * 0.25, h * 0.75],
            [w * 0.75, h * 0.75],
            [w * 0.60, h * 0.50],
            [w * 0.40, h * 0.50]
        ], dtype=np.float32)
        
        # Define destination points in the BEV image (rectangle)
        dst_w, dst_h = self.dst_size
        offset = 20  # Margin from the edges
        
        dst_points = np.array([
            [offset, dst_h - offset],  # Bottom left
            [dst_w - offset, dst_h - offset],  # Bottom right
            [dst_w - offset, offset],  # Top right
            [offset, offset]  # Top left
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        return M, src_points, dst_points
    
    def image_to_bev(self, image):
        """
        Transform image to BEV using inverse perspective mapping
        
        Args:
            image: Input camera image
            
        Returns:
            BEV image
        """
        # Get transform if not already computed
        if self.M is None:
            self.M, self.src_points, self.dst_points = self.estimate_transform(image)
        
        # Apply perspective transform
        bev_image = cv2.warpPerspective(
            image, self.M, self.dst_size, 
            flags=cv2.INTER_LINEAR
        )
        
        return bev_image
    
    def draw_source_region(self, image):
        """
        Draw the source region on the original image
        
        Args:
            image: Original image
            
        Returns:
            Image with source region drawn
        """
        if self.src_points is None:
            _, self.src_points, _ = self.estimate_transform(image)
            
        img_with_region = image.copy()
        
        # Draw the trapezoid
        points = self.src_points.astype(np.int32)
        cv2.polylines(img_with_region, [points], True, (0, 255, 0), 2)
        
        # Add a label
        cv2.putText(img_with_region, "IPM Source Region", 
                   (points[0][0], points[0][1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return img_with_region
    
    def add_grid_to_bev(self, bev_image, grid_size=5.0):
        """
        Add a grid to the BEV image for better spatial context
        
        Args:
            bev_image: BEV image to add grid to
            grid_size: Grid cell size in meters
            
        Returns:
            BEV image with grid
        """
        img_with_grid = bev_image.copy()
        h, w = img_with_grid.shape[:2]
        
        # Calculate grid interval in pixels
        x_interval = int(grid_size / self.x_resolution)
        y_interval = int(grid_size / self.y_resolution)
        
        # Draw vertical grid lines
        for x in range(0, w, x_interval):
            cv2.line(img_with_grid, (x, 0), (x, h), (50, 50, 50), 1)
        
        # Draw horizontal grid lines
        for y in range(0, h, y_interval):
            cv2.line(img_with_grid, (0, y), (w, y), (50, 50, 50), 1)
        
        # Calculate the vehicle position (typically center bottom of the BEV)
        vehicle_x = w // 2
        vehicle_y = h - 20
        
        # Draw the vehicle
        cv2.circle(img_with_grid, (vehicle_x, vehicle_y), 5, (0, 0, 255), -1)
        
        # Add distance markings
        for dist in range(int(grid_size), int(self.x_range[1]), int(grid_size)):
            # Convert distance to pixel position
            y_pos = h - int((dist / (self.x_range[1] - self.x_range[0])) * h)
            cv2.line(img_with_grid, (0, y_pos), (w, y_pos), (70, 70, 70), 1)
            cv2.putText(img_with_grid, f"{dist}m", (10, y_pos - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add title
        cv2.putText(img_with_grid, "Bird's Eye View (IPM)", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img_with_grid
    
    def project_bboxes_to_bev(self, bev_image, bboxes=None, distances=None, class_names=None):
        """
        Create a clean BEV representation without object boxes
        
        Args:
            bev_image: BEV image to return
            bboxes: Not used in this version
            distances: Not used in this version
            class_names: Not used in this version
            
        Returns:
            BEV image without any projected bounding boxes
        """
        # Simply return the original BEV image without adding any boxes
        return bev_image