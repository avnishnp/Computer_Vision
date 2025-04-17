import cv2
import glob
import numpy as np
import time
import queue
import carla
import sys
import argparse
import random
import os
from monocular_slam.display import Display
from monocular_slam.extractor import Frame, denormalize, match_frames, add_ones
from monocular_slam.pointmap import Map, Point

# Camera intrinsics
W, H = 1920 // 2, 1080 // 2
F = 450
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
Kinv = np.linalg.inv(K)

display = Display(1920, 1080)
mapp = Map()
mapp.create_viewer()

# Image queues to store frames from CARLA
main_image_queue = queue.Queue()
third_person_queue = queue.Queue()

# Open a file to save poses
pose_file = open("poses.txt", "w")
pose_file.write("# timestamp tx ty tz qx qy qz qw\n")

def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret

def process_frame(img, timestamp=None):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    # previous frame f2 to the current frame f1.
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    # print(f"=------------Rt {Rt}")

    f1.pose = np.dot(Rt, f2.pose)

    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), 2, (77, 243, 255))
        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0))
        cv2.circle(img, (u2, v2), 2, (204, 77, 255))

    # Use SDL2 display
    display.paint(img)

    mapp.display()
    mapp.display_image(img)
    
    # Save the pose to file
    save_pose(f1.pose, timestamp if timestamp else frame.id)

def save_pose(pose, timestamp):
    """
    Save the camera pose to a file in TUM RGB-D dataset format:
    timestamp tx ty tz qx qy qz qw
    """
    # Extract translation components
    tx = pose[0, 3]
    ty = pose[1, 3]
    tz = pose[2, 3]
    
    # Convert rotation matrix to quaternion
    # This is a simple implementation - there are better ways to handle edge cases
    R = pose[:3, :3]
    
    # Calculate quaternion from rotation matrix
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    
    # Write to file
    pose_file.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
    pose_file.flush()  # Ensure it's written immediately

# Callback function for the main camera
def main_camera_callback(image, data_queue):
    # Convert CARLA raw image to OpenCV format
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # RGBA to RGB
    array = array[:, :, ::-1]  # RGB to BGR (for OpenCV)
    
    # Put the image in the queue with timestamp
    timestamp = image.timestamp
    data_queue.put((array.copy(), timestamp))  # Add timestamp

# Callback function for the third-person camera
def third_person_callback(image, data_queue):
    # Convert CARLA raw image to OpenCV format
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # RGBA to RGB
    array = array[:, :, ::-1]  # RGB to BGR (for OpenCV)
    
    # Put the image in the queue
    data_queue.put(array.copy())

def get_vehicle_speed(vehicle):
    """Calculate the speed of the vehicle in m/s"""
    velocity = vehicle.get_velocity()
    return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

def is_at_traffic_light(vehicle, world):
    """Check if the vehicle is at a traffic light and return the state"""
    # Get the traffic light state affecting the vehicle
    traffic_light = vehicle.get_traffic_light()
    
    if traffic_light is not None:
        # Get the state of the traffic light
        state = traffic_light.get_state()
        
        # Check if we are in a red or yellow light
        if state == carla.TrafficLightState.Red:
            return "Red"
        elif state == carla.TrafficLightState.Yellow:
            return "Yellow"
        elif state == carla.TrafficLightState.Green:
            return "Green"
    
    # Check if we are approaching a traffic light
    vehicle_location = vehicle.get_location()
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    
    for light in traffic_lights:
        light_location = light.get_location()
        # Calculate distance to traffic light
        distance = np.sqrt((vehicle_location.x - light_location.x)**2 + 
                           (vehicle_location.y - light_location.y)**2)
        
        # If close to a traffic light (within 25 meters) and it's red
        if distance < 25 and light.get_state() == carla.TrafficLightState.Red:
            # Check if we are facing the traffic light
            forward_vector = vehicle.get_transform().get_forward_vector()
            direction_to_light = carla.Vector3D(
                light_location.x - vehicle_location.x,
                light_location.y - vehicle_location.y,
                light_location.z - vehicle_location.z
            )
            
            # Normalize direction vector
            direction_length = np.sqrt(direction_to_light.x**2 + direction_to_light.y**2 + direction_to_light.z**2)
            if direction_length > 0:
                direction_to_light.x /= direction_length
                direction_to_light.y /= direction_length
                direction_to_light.z /= direction_length
                
                # Calculate dot product to check if we're facing the light
                dot_product = forward_vector.x * direction_to_light.x + \
                              forward_vector.y * direction_to_light.y
                
                # If we're facing the traffic light (dot product > 0)
                if dot_product > 0:
                    return "Approaching Red"
    
    return None  # Not at a traffic light

def main():
    argparser = argparse.ArgumentParser(description='CARLA SLAM Client')
    argparser.add_argument(
        '--host',
        default='127.0.0.1',
        help='IP of the CARLA server (default: 127.0.0.1)')
    argparser.add_argument(
        '--port',
        default=2000,
        type=int,
        help='TCP port of CARLA server (default: 2000)')
    argparser.add_argument(
        '--spawn_idx',
        default=-1,
        type=int,
        help='Spawn point index (default: -1 for random)')
    argparser.add_argument(
        '--ignore_lights',
        action='store_true',
        help='Ignore traffic lights')
    args = argparser.parse_args()

    # Connect to CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)  # Increased timeout for better reliability

    # At the beginning of your simulation
    client.start_recorder("recording.log")

    try:
        # Use the current world
        world = client.get_world()
        print(f"Using current world: {world.get_map().name}")
        
        # Set synchronous mode for consistent frames
        settings = world.get_settings()
        sync_mode = settings.synchronous_mode
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        
        # Get blueprint library
        blueprint_library = world.get_blueprint_library()
        
        # Get all available vehicle blueprints
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        if len(vehicle_blueprints) == 0:
            print("Error: No vehicle blueprints available")
            return
        
        # Try to find a valid vehicle blueprint
        vehicle_bp = None
        for vehicle_name in ['vehicle.audi.tt', 'vehicle.mini.cooperst', 'vehicle.tesla.model3', 'vehicle.dodge.charger']:
            try:
                vehicle_bp = blueprint_library.find(vehicle_name)
                if vehicle_bp is not None:
                    print(f"Selected vehicle: {vehicle_bp.id}")
                    break
            except:
                continue
                
        # Fallback to first available vehicle
        if vehicle_bp is None:
            try:
                # Get first vehicle safely
                for vbp in vehicle_blueprints:
                    vehicle_bp = vbp
                    print(f"Fallback to first vehicle: {vehicle_bp.id}")
                    break
            except:
                print("Error getting vehicle blueprint")
                return
        
        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        if len(spawn_points) == 0:
            print("Error: No spawn points available")
            return
        
        print(f"Found {len(spawn_points)} potential spawn points")
        
        # Choose a spawn point
        if args.spawn_idx >= 0 and args.spawn_idx < len(spawn_points):
            spawn_point = spawn_points[args.spawn_idx]
            print(f"Using specified spawn point {args.spawn_idx}")
        else:
            spawn_idx = random.randint(0, min(10, len(spawn_points)-1))
            spawn_point = spawn_points[spawn_idx]
            print(f"Using spawn point {spawn_idx} (random selection from first 10)")
        
        # Spawn vehicle with multiple attempts
        vehicle = None
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Slightly raise the spawn point to avoid collision with ground
                adjusted_spawn = carla.Transform(
                    carla.Location(
                        x=spawn_point.location.x,
                        y=spawn_point.location.y,
                        z=spawn_point.location.z + 0.2
                    ),
                    spawn_point.rotation
                )
                
                vehicle = world.spawn_actor(vehicle_bp, adjusted_spawn)
                print(f"Vehicle spawned at {adjusted_spawn.location} (attempt {attempt+1})")
                break
            except Exception as e:
                print(f"Failed to spawn vehicle at attempt {attempt+1}: {e}")
                # Try a different spawn point
                spawn_point = random.choice(spawn_points)
                if attempt == max_attempts - 1:
                    print("Maximum spawn attempts reached. Exiting.")
                    return
        
        # Set up the traffic manager and enable autopilot
        print("Setting up traffic manager...")
        traffic_manager = client.get_trafficmanager(8000)  # Port 8000
        traffic_manager.set_synchronous_mode(True)
        
        # Configure traffic manager
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)  # Safe distance
        traffic_manager.global_percentage_speed_difference(-20)  # Drive 20% faster
        
        # Optionally ignore traffic lights
        if args.ignore_lights:
            print("Ignoring traffic lights")
            traffic_manager.ignore_lights_percentage(vehicle, 100)
            
        # Enable autopilot with traffic manager
        print("Enabling vehicle autopilot")
        vehicle.set_autopilot(True, traffic_manager.get_port())
        
        # First-person camera setup (for SLAM)
        fp_camera_bp = blueprint_library.find('sensor.camera.rgb')
        fp_camera_bp.set_attribute('image_size_x', str(W*2))
        fp_camera_bp.set_attribute('image_size_y', str(H*2))
        fp_camera_bp.set_attribute('fov', '90')
        
        fp_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        fp_camera = world.spawn_actor(fp_camera_bp, fp_camera_transform, attach_to=vehicle)
        
        # Third-person camera setup - try a very high position
        tp_camera_bp = blueprint_library.find('sensor.camera.rgb')
        tp_camera_bp.set_attribute('image_size_x', str(W*2))
        tp_camera_bp.set_attribute('image_size_y', str(H*2))
        tp_camera_bp.set_attribute('fov', '110')  # Very wide FOV
        
        # Position the camera very high up
        tp_camera_transform = carla.Transform(
            carla.Location(x=0, z=30, y=0),  # Directly above
            carla.Rotation(pitch=-90)  # Looking straight down
        )
        
        # Create a window for the third-person view
        cv2.namedWindow('Third-Person View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Third-Person View', W, H)
        
        try:
            print("Trying to spawn third-person camera...")
            tp_camera = world.spawn_actor(tp_camera_bp, tp_camera_transform, attach_to=vehicle)
            print("Third-person camera spawned successfully")
            has_tp_camera = True
        except Exception as e:
            print(f"Failed to spawn third-person camera: {e}")
            print("Continuing without third-person camera")
            tp_camera = None
            has_tp_camera = False
        
        # Register callbacks
        fp_camera.listen(lambda image: main_camera_callback(image, main_image_queue))
        if has_tp_camera:
            tp_camera.listen(lambda image: third_person_callback(image, third_person_queue))
        
        print("CARLA setup complete. Running SLAM...")
        print(f"Poses will be saved to: {os.path.abspath('poses.txt')}")
        
        # Wait for autopilot to initialize
        print("Waiting for autopilot to initialize...")
        for _ in range(10):  # Wait for 10 ticks
            world.tick()
            time.sleep(0.05)
            
        # Variables for the main loop
        frame_count = 0
        min_speed_threshold = 0.5  # m/s
        autopilot_issue_counter = 0
        traffic_light_wait_time = 0  # Track how long we've been at a red light
        
        try:
            # Main loop
            while True:
                # Advance simulation
                world.tick()
                frame_count += 1
                
                # Get the current speed
                speed = get_vehicle_speed(vehicle)
                
                # Print speed occasionally to reduce output spam
                if frame_count % 20 == 0:
                    print(f"Vehicle speed: {speed:.2f} m/s")
                
                # Check if vehicle is stopped
                if speed < 0.1:
                    # Check if we're at a traffic light
                    light_state = is_at_traffic_light(vehicle, world)
                    
                    if light_state == "Red" or light_state == "Yellow" or light_state == "Approaching Red":
                        # We're at a red or yellow light - this is normal
                        traffic_light_wait_time += 1
                        if traffic_light_wait_time % 40 == 0:  # Print every 2 seconds
                            print(f"Waiting at {light_state} traffic light... ({traffic_light_wait_time * 0.05:.1f}s)")
                    else:
                        # Not at a traffic light, might be stuck
                        autopilot_issue_counter += 1
                        
                        if autopilot_issue_counter >= 100:  # If stuck for 5 seconds
                            print("Vehicle appears to be stuck (not at traffic light).")
                            print("Teleporting to a new position...")
                            
                            # Choose a new spawn point
                            new_spawn = random.choice(spawn_points)
                            vehicle.set_transform(new_spawn)
                            print(f"Teleported to {new_spawn.location}")
                            
                            # Reset counters
                            autopilot_issue_counter = 0
                            traffic_light_wait_time = 0
                else:
                    # Reset counters when moving
                    autopilot_issue_counter = 0
                    traffic_light_wait_time = 0
                
                # Get frames from both cameras
                main_data = None
                
                if not main_image_queue.empty():
                    main_data = main_image_queue.get()
                    main_frame = main_data[0]  # Image is first element
                    timestamp = main_data[1]  # Timestamp is second element
                
                # Try to get third-person view if available
                if has_tp_camera and not third_person_queue.empty():
                    tp_frame = third_person_queue.get()
                    # Display the third-person view in a window instead of saving to file
                    cv2.imshow('Third-Person View', tp_frame)
                
                # Process frames if available AND the vehicle is moving OR at a traffic light
                light_state = is_at_traffic_light(vehicle, world)  # Make sure light_state is defined
                if main_data is not None:
                    if speed > min_speed_threshold:
                        process_frame(main_frame, timestamp)
                    elif light_state in ["Red", "Yellow", "Approaching Red"] and traffic_light_wait_time % 20 == 0:
                        # At a traffic light - generate some frames occasionally while stopped
                        # This allows capturing the static scene while waiting
                        print(f"Processing frame while stopped at {light_state} light")
                        process_frame(main_frame, timestamp)
                    elif frame_count % 20 == 0:
                        print(f"Skipping SLAM at low speed: {speed:.2f} m/s")
                
                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Cleanup
            fp_camera.stop()
            if has_tp_camera:
                tp_camera.stop()
            vehicle.destroy()
            fp_camera.destroy()
            if has_tp_camera:
                tp_camera.destroy()
            
            # Close pose file
            pose_file.close()
            print(f"Poses saved to: {os.path.abspath('poses.txt')}")
            
            # Restore original settings
            settings.synchronous_mode = sync_mode
            world.apply_settings(settings)
            
    except Exception as e:
        print(f"Error: {e}")
        pose_file.close()  # Make sure to close the file even if there's an error
    
    finally:
        cv2.destroyAllWindows()
        print("SLAM terminated.")

    client.stop_recorder()
    print("Recorder stopped.")

if __name__ == "__main__":
    main()