import cv2
import numpy as np
import pickle


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

        # Create a brute-force matcher
        # The brute-force matcher compares each descriptor from the first image with every descriptor from the second image.
    matcher = cv2.BFMatcher()

    # Find the k-nearest matches
    #  The function uses the matcher's knnMatch method to find the k-nearest matches between the SIFT descriptors of the two images. 
    # It compares each descriptor from the first image with the descriptors of the second image and finds the k descriptors with the smallest distances.
    k = 2
    matches = matcher.knnMatch(des1, des2, k)

    # Apply the ratio test to filter good matches
    #  For each pair of matches (m, n), it checks if the distance of the best match m is less than 0.7 times the distance of the second best match n
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Get the matched keypoint locations
    # The function extracts the matched keypoint locations (x1 and x2) from the original keypoint location arrays (loc1 and loc2) 
    # based on the indices of the good matches.
    x1 = loc1[[m.queryIdx for m in good_matches]]
    x2 = loc2[[m.trainIdx for m in good_matches]]
    
    # Get the indices of x1 in loc1
    # returns the matched keypoint locations and their corresponding indices.
    ind1 = np.array([m.queryIdx for m in good_matches])

    return x1, x2, ind1

# The essential matrix captures the geometric relationship between two views of a scene, allowing us to recover the relative camera pose and perform 3D reconstruction.
def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """

    n = x1.shape[0]

    A = []
    for i in range(n):
        ux, uy, vx, vy = x1[i, 0], x1[i, 1], x2[i, 0], x2[i, 1]
        A.append([ux * vx, uy * vx, vx, ux * vy, uy * vy, vy, ux, uy, 1])

    A = np.stack(A)

    _, _, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)

    U, _, Vh = np.linalg.svd(E)
    D = np.eye(3)
    D[2, 2] = 0
    E = U @ D @ Vh

    return E



def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier_index : ndarray of shape (k,)
        The inlier indices
    """
    n = x1.shape[0]
    max_inlier = 0
    inlier_index = None
    E = np.eye(3)

    homogeneous1 = np.insert(x1, 2, 1, axis=1)
    homogeneous2 = np.insert(x2, 2, 1, axis=1)

    for n_step in range(ransac_n_iter):
        sample_idx = np.random.choice(n, 8)
        sampled_x1 = x1[sample_idx]
        sampled_x2 = x2[sample_idx]

        calculated_E = EstimateE(sampled_x1, sampled_x2)

        error = np.abs(np.diag(homogeneous1 @ calculated_E.T @ homogeneous2.T))

        total_inliers = np.sum(error < ransac_thr)

        if total_inliers > max_inlier:
            max_inlier = total_inliers
            E = calculated_E
            inlier_index = np.array(np.nonzero(error < ransac_thr)).reshape(-1)

    return E, inlier_index


def cvt_keypoint(kp):
    """
    convert keypoint result to numpy array
    """
    N = len(kp)
    locations = np.zeros((N, 2))
    for i in range(N):
        locations[i, :] = np.array(kp[i].pt)
    return locations

def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track_full : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    N = Im.shape[0]

    location, descriptor, num_index = [], [], [0]
    sift = cv2.SIFT_create()

    print('Extract SIFT features...')
    for i in range(N):
        kp, des = sift.detectAndCompute(Im[i], None)
        loc = cvt_keypoint(kp)

        location.append(loc)
        descriptor.append(des)
        num_index.append(num_index[-1] + loc.shape[0])

        print('image %d, found %d features' % (i, loc.shape[0]))

    track_full = np.empty((N, 0, 2))
    for i in range(N - 1):
        print('Build track %d....' % (i))

        nft = location[i].shape[0]
        track_i = -1 * np.ones((N, nft, 2))

        location1 = location[i]
        descriptor1 = descriptor[i]

        for j in range(i + 1, N):
            location2 = location[j]
            descriptor2 = descriptor[j]

            x1, x2, index1 = MatchSIFT(location1, descriptor1, location2, descriptor2)
            print('Found %d matched pairs between image %d and %d' % (x1.shape[0], i, j))

            normalized1 = np.insert(x1, 2, 1, axis=1) @ np.linalg.inv(K).T
            normalized2 = np.insert(x2, 2, 1, axis=1) @ np.linalg.inv(K).T

            normalized1 = normalized1[:, :2]
            normalized2 = normalized2[:, :2]

            E, inlier_index = EstimateE_RANSAC(normalized1, normalized2, 500, 0.003)
            print('%d matched pairs remains after essential matrix estimation' % (inlier_index.shape[0]))

            track_index = index1[inlier_index]

            track_i[i, track_index, :] = normalized1[inlier_index]
            track_i[j, track_index, :] = normalized2[inlier_index]

        mask = np.sum(track_i[i], axis=1) != -2
        track_i = track_i[:, mask, :]
        print('Adding %d feature matches from image %d into track' % (track_i.shape[1], i))
        track_full = np.concatenate([track_full, track_i], axis=1)

    # outfile = open('track.pkl', 'wb')
    # pickle.dump(track_full, outfile)
    # outfile.close()
    return track_full