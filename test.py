import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from feature import EstimateE,EstimateE_RANSAC
from pnp import PnP_RANSAC
# from camera_pose import Triangulation
from camera_pose import GetCameraPoseFromE, Triangulation,EvaluateCheirality

def MatchSIFT(loc1, des1, loc2, des2):
    # Function implementation goes here...
    matcher = cv2.BFMatcher()
    k = 2
    matches = matcher.knnMatch(des1, des2, k)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    x1 = loc1[[m.queryIdx for m in good_matches]]
    x2 = loc2[[m.trainIdx for m in good_matches]]
    ind1 = np.array([m.queryIdx for m in good_matches])

    return x1, x2, ind1

def draw_epipolar_lines(image, lines, points):
    image_copy = image.copy()
    for line, point in zip(lines, points):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = int(0), int(-line[2] / line[1])
        x1, y1 = int(image.shape[1]), int(-(line[2] + line[0] * image.shape[1]) / line[1])
        image_copy = cv2.line(image_copy, (x0, y0), (x1, y1), color, 1)
        image_copy = cv2.circle(image_copy, (int(point[0]), int(point[1])), 5, color, -1)
    return image_copy


# Step 1: Load the images
image1 = cv2.imread('im/image0000001.jpg')
image2 = cv2.imread('im/image0000002.jpg')

# Step 2: Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Step 3: Detect keypoints and compute descriptors
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Step 4: Call the MatchSIFT function
# The pt attribute of a KeyPoint object stores the (x, y) coordinates of the keypoint in the image.
x1, x2, ind1 = MatchSIFT(np.float32([kp.pt for kp in keypoints1]), descriptors1,
                         np.float32([kp.pt for kp in keypoints2]), descriptors2)


# Step 4: Retrieve matched keypoint locations and indices
matched_locations_image1 = x1
matched_locations_image2 = x2
matched_indices_image1 = ind1

# Step 5: Convert keypoints to DMatch format
# DMatch is a class in OpenCV that represents a match between two keypoints. It is used to store information about the correspondence between keypoints in different images.
'''
The DMatch class has the following attributes:

queryIdx: The index of the keypoint in the query (first) image.
trainIdx: The index of the keypoint in the train (second) image.
distance: The distance between the descriptors of the matched keypoints. This distance can be used to measure the similarity between keypoints.
'''
matches = [cv2.DMatch(idx, idx, 0) for idx in range(len(ind1))]
# print(matches)
# print(matches)

# Step 6: Visualize the matches
# cv2.drawMatches() is a function in OpenCV that is used to draw lines connecting the matched keypoints between two images. 
# It visualizes the correspondence between keypoints in a visually informative way.
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)

# Convert BGR image to RGB
matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)

# Step 7: Estimate the essential matrix
E = EstimateE(matched_locations_image1, matched_locations_image2)


# Step 6: Estimate essential matrix using RANSAC
ransac_n_iter = 1000
ransac_thr = 1.1
E_ransac, inlier_index = EstimateE_RANSAC(matched_locations_image1, matched_locations_image2, ransac_n_iter, ransac_thr)

# Step 8: Compute the fundamental matrix
F, mask = cv2.findFundamentalMat(matched_locations_image1, matched_locations_image2, cv2.FM_RANSAC, 0.1, 0.99)

# Step 9: Compute the epipolar lines
# lines1 = cv2.computeCorrespondEpilines(matched_locations_image1.reshape(-1, 1, 2), 1, F)
# lines1 = lines1.reshape(-1, 3)
# lines2 = cv2.computeCorrespondEpilines(matched_locations_image2.reshape(-1, 1, 2), 2, F)
# lines2 = lines2.reshape(-1, 3)
lines1 = cv2.computeCorrespondEpilines(matched_locations_image1.reshape(-1, 1, 2), 1, E_ransac)
lines1 = lines1.reshape(-1, 3)
lines2 = cv2.computeCorrespondEpilines(matched_locations_image2.reshape(-1, 1, 2), 2, E_ransac)
lines2 = lines2.reshape(-1, 3)

# Step 10: Draw the epipolar lines on the images
image1_with_lines = draw_epipolar_lines(image1, lines2, matched_locations_image1)
image2_with_lines = draw_epipolar_lines(image2, lines1, matched_locations_image2)

# Step 9: Obtain camera poses from the essential matrix
R_set, C_set = GetCameraPoseFromE(E_ransac)

# Print the camera poses
for i in range(len(R_set)):
    print("Camera Pose", i+1)
    print("Rotation Matrix:")
    print(R_set[i])
    print("Camera Center:")
    print(C_set[i])
    print()



# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the camera coordinate frames
for i in range(len(R_set)):
    R = R_set[i]
    C = C_set[i]

    # Define the camera coordinate axes
    axes_length = 1.0
    x_axis = axes_length * R[:, 0]
    y_axis = axes_length * R[:, 1]
    z_axis = axes_length * R[:, 2]

    # Plot the camera coordinate axes with an offset
    offset = 2.0 * axes_length * (i // 2)
    ax.quiver(C[0], C[1], C[2] + offset, x_axis[0], x_axis[1], x_axis[2], color='r', length=axes_length)
    ax.quiver(C[0], C[1], C[2] + offset, y_axis[0], y_axis[1], y_axis[2], color='g', length=axes_length)
    ax.quiver(C[0], C[1], C[2] + offset, z_axis[0], z_axis[1], z_axis[2], color='b', length=axes_length)

    # Plot the camera centers with an offset
    ax.scatter(C[0], C[1], C[2] + offset, c='r', marker='o')
    ax.text(C[0], C[1], C[2] + offset, f'Camera {i+1}', color='r')

# Set plot limits and labels
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# Step 15: Perform triangulation
n = matched_locations_image1.shape[0]
track1 = np.concatenate((matched_locations_image1, np.ones((n, 1))), axis=1)
track2 = np.concatenate((matched_locations_image2, np.ones((n, 1))), axis=1)

P = np.zeros((2, 3, 4))

P[0] = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]])
P[1] = np.hstack([R, -(R @ C).reshape((3, 1))])

X = Triangulation(P[0], P[1], track1, track2)

# Step 16: Evaluate cheirality
valid_index = EvaluateCheirality(P[0], P[1], X)

# Step 17: Filter valid 3D points
valid_points = X[valid_index]

# Step 18: Visualize the valid 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], c='b', marker='o')
# Step 18: Visualize individual camera configurations and point clouds
for i in range(len(R_set)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the camera configuration for the current camera
    R = R_set[i]
    C = C_set[i]

    # Define the camera coordinate axes
    axes_length = 1.0
    x_axis = axes_length * R[:, 0]
    y_axis = axes_length * R[:, 1]
    z_axis = axes_length * R[:, 2]

    # Plot the camera coordinate axes with an offset
    offset = 2.0 * axes_length * (i // 2)
    ax.quiver(C[0], C[1], C[2] + offset, x_axis[0], x_axis[1], x_axis[2], color='r', length=axes_length)
    ax.quiver(C[0], C[1], C[2] + offset, y_axis[0], y_axis[1], y_axis[2], color='g', length=axes_length)
    ax.quiver(C[0], C[1], C[2] + offset, z_axis[0], z_axis[1], z_axis[2], color='b', length=axes_length)

    # Plot the camera centers with an offset
    ax.scatter(C[0], C[1], C[2] + offset, c='r', marker='o')
    ax.text(C[0], C[1], C[2] + offset, f'Camera {i+1}', color='r')
    
    # Filter valid points belonging to the current camera configuration
    valid_points_i = valid_points[i::len(R_set)]
    
    if len(valid_points_i) > 0:
        # Extract x, y, and z coordinates of valid points
        x = valid_points_i[:, 0]
        y = valid_points_i[:, 1]
        z = valid_points_i[:, 2]
        
        # Plot the valid 3D points for the current camera configuration
        ax.scatter(x, y, z, c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Camera {i+1} Point Cloud")
    plt.show()

    print(f"Number of Valid Points for Camera {i+1}: {len(valid_points_i)}")

# Step 11: Display the images with epipolar lines
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image1_with_lines)
plt.title("Image 1 with Epipolar Lines")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image2_with_lines)
plt.title("Image 2 with Epipolar Lines")
plt.axis('off')

# Display the matched image and inlier image using matplotlib
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(matched_image_rgb)
plt.title("All Matches")
plt.axis('off')

print("Essential Matrix:")
print(E)
print("RANSAC Essential Matrix:")
print(E_ransac)
plt.axis('off')
plt.show()
