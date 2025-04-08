import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d

class EnhancedBEVVisualizer:
    """
    Enhanced Bird's Eye View visualizer with more prominent bounding boxes,
    clear class labels, and ground truth visualization
    """
    def __init__(self, x_range=(-20, 20), y_range=(-20, 20), resolution=0.1, height_threshold=(-2, 1)):
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.height_threshold = height_threshold
        
        # Calculate dimensions of the BEV image
        self.x_size = int((x_range[1] - x_range[0]) / resolution)
        self.y_size = int((y_range[1] - y_range[0]) / resolution)
        
        # Create color map for depth visualization
        self.cmap = plt.cm.get_cmap('jet')
        
        # Class colors dictionary (BGR format for OpenCV)
        self.class_colors = {
            0: (0, 0, 255),    # Person/Pedestrian: Red
            1: (0, 255, 255),  # Bicycle/Cyclist: Yellow
            2: (0, 255, 0),    # Car/Van: Green
            3: (255, 255, 0),  # Motorcycle: Cyan
            4: (255, 0, 255),  # Airplane: Magenta
            5: (255, 165, 0),  # Bus: Orange
            6: (138, 43, 226), # Train/Tram: Purple
            7: (255, 0, 0),    # Truck: Blue
            8: (255, 192, 203),# Boat: Pink
            9: (255, 255, 255) # Traffic light: White
        }
        
        # Class names dictionary (updated to match KITTI labels)
        self.class_names = {
            0: 'person',    # Maps to Pedestrian and Person_sitting
            1: 'cyclist',   # Maps to Cyclist
            2: 'car',       # Maps to Car and Van
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'tram',      # Maps to Tram
            7: 'truck',     # Maps to Truck
            8: 'boat',
            9: 'traffic light'
        }
        
        # KITTI specific class mapping
        self.kitti_to_class = {
            'Car': 2,
            'Pedestrian': 0,
            'Cyclist': 1,
            'Truck': 7,
            'Van': 2,
            'Person_sitting': 0,
            'Tram': 6,
            'Misc': -1
        }
    
    def point_to_pixel(self, x, y):
        """Convert from LiDAR coordinates to BEV pixel coordinates"""
        # Map LiDAR x,y coordinates to pixel coordinates
        pixel_x = int((x - self.x_range[0]) / self.resolution)
        pixel_y = int(self.y_size - (y - self.y_range[0]) / self.resolution)
        
        # Ensure pixel coordinates are within bounds
        pixel_x = max(0, min(self.x_size - 1, pixel_x))
        pixel_y = max(0, min(self.y_size - 1, pixel_y))
        
        return pixel_x, pixel_y
    
    def create_base_bev_image(self, point_cloud, color_by='height'):
        """Create a base BEV image with LiDAR points and grid"""
        # Initialize empty BEV image with dark gray background
        bev_image = np.ones((self.y_size, self.x_size, 3), dtype=np.uint8) * 20
        
        # Add a grid pattern to the background
        grid_spacing = int(1.0 / self.resolution)  # 1-meter grid
        
        # Draw darker grid lines
        for i in range(0, self.x_size, grid_spacing):
            cv2.line(bev_image, (i, 0), (i, self.y_size), (30, 30, 30), 1)
        for i in range(0, self.y_size, grid_spacing):
            cv2.line(bev_image, (0, i), (self.x_size, i), (30, 30, 30), 1)
        
        # Draw major grid lines (every 5 meters)
        for i in range(0, self.x_size, grid_spacing * 5):
            cv2.line(bev_image, (i, 0), (i, self.y_size), (40, 40, 40), 2)
        for i in range(0, self.y_size, grid_spacing * 5):
            cv2.line(bev_image, (0, i), (self.x_size, i), (40, 40, 40), 2)
        
        # Draw distance rings
        origin_x, origin_y = self.point_to_pixel(0, 0)
        for radius in [5, 10, 15, 20, 25]:
            pixel_radius = int(radius / self.resolution)
            cv2.circle(bev_image, (origin_x, origin_y), pixel_radius, (50, 50, 50), 1)
            # Add distance label at 0 degrees (straight ahead)
            x_pos, y_pos = self.point_to_pixel(radius, 0)
            cv2.putText(bev_image, f"{radius}m", (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Draw coordinate axes
        # X-axis (forward)
        end_x, end_y = self.point_to_pixel(10, 0)
        cv2.line(bev_image, (origin_x, origin_y), (end_x, end_y), (0, 0, 200), 2)
        cv2.putText(bev_image, "X", (end_x + 5, end_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
        
        # Y-axis (left)
        end_x, end_y = self.point_to_pixel(0, 10)
        cv2.line(bev_image, (origin_x, origin_y), (end_x, end_y), (0, 200, 0), 2)
        cv2.putText(bev_image, "Y", (end_x, end_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        
        # Draw vehicle outline (approximately 4.5m x 1.8m)
        vehicle_length = 4.5  # meters
        vehicle_width = 1.8   # meters
        
        # Calculate corners for vehicle outline
        front_left = self.point_to_pixel(vehicle_length/2, vehicle_width/2)
        front_right = self.point_to_pixel(vehicle_length/2, -vehicle_width/2)
        rear_left = self.point_to_pixel(-vehicle_length/2, vehicle_width/2)
        rear_right = self.point_to_pixel(-vehicle_length/2, -vehicle_width/2)
        
        # Draw vehicle as a filled polygon
        vehicle_corners = np.array([front_left, front_right, rear_right, rear_left])
        cv2.fillPoly(bev_image, [vehicle_corners], (120, 120, 120))
        
        # Add directional marker
        center = self.point_to_pixel(0, 0)
        front_center = self.point_to_pixel(vehicle_length/2, 0)
        cv2.line(bev_image, center, front_center, (200, 200, 200), 2)
        
        # Filter points by height
        if self.height_threshold and len(point_cloud) > 0:
            height_mask = (point_cloud[:, 2] >= self.height_threshold[0]) & (point_cloud[:, 2] <= self.height_threshold[1])
            point_cloud = point_cloud[height_mask]
        
        # Draw LiDAR points
        if len(point_cloud) > 0:
            # Calculate values for coloring (by height or distance)
            if color_by == 'height':
                values = point_cloud[:, 2]
                vmin, vmax = self.height_threshold
            else:  # distance
                values = np.sqrt(point_cloud[:, 0]**2 + point_cloud[:, 1]**2)
                vmin, vmax = 0, 50
            
            # Normalize values for coloring
            norm_values = np.clip((values - vmin) / (vmax - vmin), 0, 1)
            
            # Draw each point
            for i in range(len(point_cloud)):
                x, y = point_cloud[i, 0], point_cloud[i, 1]
                
                # Skip points outside range
                if x < self.x_range[0] or x >= self.x_range[1] or y < self.y_range[0] or y >= self.y_range[1]:
                    continue
                
                pixel_x, pixel_y = self.point_to_pixel(x, y)
                color = self.cmap(norm_values[i])[:3]
                
                # Draw point
                cv2.circle(bev_image, (pixel_x, pixel_y), 1, 
                          tuple(map(lambda x: int(x * 255), color)), -1)
        
        return bev_image
    
    def draw_detection_box(self, bev_image, center, dimensions, class_id=2, distance=None, 
                          thickness=2, is_ground_truth=False):
        """
        Draw a single detection box in the BEV image with clear visibility
        
        Args:
            bev_image: BEV image to draw on
            center: (x, y, z) center of the object
            dimensions: (length, width, height) dimensions of the object
            class_id: Class ID of the object
            distance: Distance to the object in meters
            thickness: Line thickness
            is_ground_truth: Whether this is a ground truth box (different style)
            
        Returns:
            BEV image with drawn box
        """
        # Get class color and name
        color = self.class_colors.get(class_id, (0, 255, 255))  # Default to yellow
        class_name = self.class_names.get(class_id, f"class_{class_id}")
        
        # For ground truth, we'll use white borders with the class color
        if is_ground_truth:
            border_color = (255, 255, 255)  # White
            fill_color = color
            line_type = cv2.LINE_AA  # Anti-aliased line
        else:
            border_color = color
            fill_color = color
            line_type = cv2.LINE_AA
        
        # Unpack dimensions
        length, width, height = dimensions
        
        # Create box corners (top view)
        corners = np.array([
            [length/2, width/2, 0],   # Front right
            [length/2, -width/2, 0],  # Front left
            [-length/2, -width/2, 0], # Rear left
            [-length/2, width/2, 0],  # Rear right
        ])
        
        # Translate corners to world coordinates
        corners += center[:3]
        
        # Convert to pixel coordinates
        pixel_corners = []
        for corner in corners:
            pixel_x, pixel_y = self.point_to_pixel(corner[0], corner[1])
            pixel_corners.append((pixel_x, pixel_y))
        
        # Create a copy for transparent overlay
        overlay = bev_image.copy()
        
        # Fill the box with semi-transparent color
        cv2.fillPoly(overlay, [np.array(pixel_corners)], fill_color)
        
        # Blend with original image
        alpha = 0.2 if is_ground_truth else 0.4  # More transparent for ground truth
        cv2.addWeighted(overlay, alpha, bev_image, 1 - alpha, 0, bev_image)
        
        # Draw the box outline
        if is_ground_truth:
            # Draw dashed line for ground truth
            for i in range(4):
                start_point = pixel_corners[i]
                end_point = pixel_corners[(i + 1) % 4]
                
                # Create dashed line effect
                pt1 = np.array(start_point)
                pt2 = np.array(end_point)
                dist = np.linalg.norm(pt2 - pt1)
                pts = []
                
                # Create points for dashed line
                for j in range(0, int(dist), 8):
                    r = j / dist
                    point = tuple(map(int, pt1 + r * (pt2 - pt1)))
                    pts.append(point)
                
                # Draw dashed line segments
                for j in range(0, len(pts) - 1, 2):
                    if j + 1 < len(pts):
                        cv2.line(bev_image, pts[j], pts[j+1], border_color, thickness, line_type)
        else:
            # Draw solid line for predictions
            for i in range(4):
                start_point = pixel_corners[i]
                end_point = pixel_corners[(i + 1) % 4]
                cv2.line(bev_image, start_point, end_point, border_color, thickness, line_type)
        
        # Add label with class name and distance
        pixel_x, pixel_y = self.point_to_pixel(center[0], center[1])
        
        if distance is not None:
            if is_ground_truth:
                label = f"GT: {class_name} {distance:.1f}m"
            else:
                label = f"{class_name} {distance:.1f}m"
        else:
            if is_ground_truth:
                label = f"GT: {class_name}"
            else:
                label = class_name
        
        # Draw text with background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
        
        # Position text above the box
        text_x = pixel_x - text_size[0] // 2
        text_y = pixel_y - 10
        
        # Ensure text is within image bounds
        text_x = max(5, min(self.x_size - text_size[0] - 5, text_x))
        text_y = max(15, text_y)
        
        # Draw text background
        cv2.rectangle(bev_image, 
                     (text_x - 2, text_y - text_size[1] - 2),
                     (text_x + text_size[0] + 2, text_y + 2),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(bev_image, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return bev_image
    
    def estimate_3d_box(self, detection, distance=None):
        """
        Estimate 3D box parameters from a 2D detection and distance
        
        Args:
            detection: [x1, y1, x2, y2, conf, cls] from YOLO detection
            distance: Distance to the object, or None to estimate
            
        Returns:
            center (x,y,z), dimensions (l,w,h), and yaw angle
        """
        # Default class dimensions (length, width, height) in meters
        class_dimensions = {
            0: (0.6, 0.6, 1.7),  # Person/Pedestrian
            1: (1.8, 0.7, 1.0),  # Bicycle/Cyclist
            2: (4.5, 1.8, 1.5),  # Car/Van
            3: (2.0, 0.8, 1.2),  # Motorcycle
            4: (12.0, 3.0, 3.0), # Airplane
            5: (11.0, 2.6, 3.0), # Bus
            6: (15.0, 2.5, 3.5), # Train/Tram
            7: (8.0, 2.5, 3.0),  # Truck
            8: (4.0, 2.0, 1.5),  # Boat
            9: (0.3, 0.3, 1.0)   # Traffic light
        }
        
        x1, y1, x2, y2, conf, cls = detection
        class_id = int(cls)
        
        # Use dimensions based on class
        dimensions = class_dimensions.get(class_id, (3.0, 1.5, 1.5))
        
        # Estimate distance if not provided
        if distance is None:
            # Estimate based on bounding box size
            bbox_height = y2 - y1
            # Simple inverse relationship (larger boxes are closer)
            distance = max(5.0, min(50.0, 40.0 / (bbox_height / 100.0)))
        
        # Center X is the distance
        x = distance
        
        # Calculate image width and use it to estimate lateral position (Y)
        img_width = 1280  # Default/assumed value
        bbox_center_x = (x1 + x2) / 2
        
        # Y position (positive is left)
        y_ratio = (bbox_center_x - img_width/2) / (img_width/2)
        y = -y_ratio * distance * 0.8  # Scaled by viewing angle
        
        # Z position (height above ground)
        z = -dimensions[2] / 2  # Bottom of box at ground level
        
        # Create center point and return
        center = np.array([x, y, z])
        
        return center, dimensions, 0  # 0 is the yaw angle
    
    def draw_all_detections(self, bev_image, detections, distances=None):
        """
        Draw all detected objects in the BEV image
        
        Args:
            bev_image: BEV image to draw on
            detections: List of [x1, y1, x2, y2, conf, cls] from YOLO
            distances: Optional list of distances to objects
            
        Returns:
            BEV image with all detections drawn
        """
        if distances is None:
            distances = [None] * len(detections)
        elif len(distances) < len(detections):
            distances = distances + [None] * (len(detections) - len(distances))
        
        # Draw boxes from farthest to nearest for better visibility
        detection_info = []
        for i, detection in enumerate(detections):
            distance = distances[i]
            if distance is None:
                # Estimate distance
                x1, y1, x2, y2, conf, cls = detection
                bbox_height = y2 - y1
                distance = max(5.0, min(50.0, 40.0 / (bbox_height / 100.0)))
            
            detection_info.append((detection, distance, i))
        
        # Sort by distance (farthest first)
        detection_info.sort(key=lambda x: -x[1])
        
        # Draw each detection
        for detection, distance, idx in detection_info:
            x1, y1, x2, y2, conf, cls = detection
            class_id = int(cls)
            
            # Estimate 3D box parameters
            center, dimensions, yaw = self.estimate_3d_box(detection, distances[idx])
            
            # Draw the box
            bev_image = self.draw_detection_box(
                bev_image, center, dimensions, class_id, 
                distance=distances[idx] if distances[idx] is not None else distance,
                thickness=2
            )
        
        return bev_image
    
    def add_legend(self, bev_image, show_gt=False):
        """Add a legend to the BEV image showing class colors"""
        # Create a region for the legend in the top-left
        legend_x = 10
        legend_y = 30
        
        # Add title
        cv2.putText(bev_image, "Bird's Eye View", (legend_x, legend_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add class entries - focus on the KITTI classes
        important_classes = [0, 1, 2, 7, 6]  # Person, Cyclist, Car, Truck, Tram
        
        for class_id in important_classes:
            if class_id not in self.class_names:
                continue
                
            color = self.class_colors[class_id]
            name = self.class_names[class_id]
            
            cv2.rectangle(bev_image, (legend_x, legend_y), (legend_x + 20, legend_y + 15), color, -1)
            cv2.putText(bev_image, name, (legend_x + 25, legend_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            
            legend_y += 20
        
        # Add ground truth legend if needed
        if show_gt:
            legend_y += 10
            # Draw dashed white box
            pt1 = (legend_x, legend_y)
            pt2 = (legend_x + 20, legend_y + 15)
            
            # Draw dashed outline
            for i in range(pt1[0], pt2[0], 4):
                if i + 4 <= pt2[0]:
                    cv2.line(bev_image, (i, pt1[1]), (i + 2, pt1[1]), (255, 255, 255), 1)
                    cv2.line(bev_image, (i, pt2[1]), (i + 2, pt2[1]), (255, 255, 255), 1)
            
            for i in range(pt1[1], pt2[1], 4):
                if i + 4 <= pt2[1]:
                    cv2.line(bev_image, (pt1[0], i), (pt1[0], i + 2), (255, 255, 255), 1)
                    cv2.line(bev_image, (pt2[0], i), (pt2[0], i + 2), (255, 255, 255), 1)
            
            cv2.putText(bev_image, "Ground Truth", (legend_x + 25, legend_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        return bev_image


def create_enhanced_bev(point_cloud, detections, distances=None, x_range=(-30, 30), y_range=(-20, 20)):
    """
    Create enhanced BEV visualization with prominent bounding boxes and class labels
    
    Args:
        point_cloud: Nx3 or Nx4 numpy array of LiDAR points
        detections: List of YOLO detections [x1, y1, x2, y2, conf, cls]
        distances: Optional list of distances to objects
        x_range: Range of x-coordinates to visualize (forward/backward)
        y_range: Range of y-coordinates to visualize (left/right)
        
    Returns:
        BEV visualization as a numpy image array
    """
    # Initialize the enhanced BEV visualizer
    bev_viz = EnhancedBEVVisualizer(
        x_range=x_range, 
        y_range=y_range, 
        resolution=0.1, 
        height_threshold=(-2, 1)
    )
    
    # Create base BEV image with LiDAR points
    bev_image = bev_viz.create_base_bev_image(point_cloud, color_by='height')
    
    # Draw all detections with prominent bounding boxes
    if detections is not None:
        bev_image = bev_viz.draw_all_detections(bev_image, detections, distances)
    
    # Add a legend
    bev_image = bev_viz.add_legend(bev_image)
    
    return bev_image