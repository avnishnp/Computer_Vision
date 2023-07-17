import numpy as np
import sys
from feature import EstimateE_RANSAC
from scipy.linalg import svd

def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    U, D, Vt = np.linalg.svd(E)

    R1 = U @ W @ Vt
    C1 = U[:, 2]
    if np.linalg.det(R1) < 0:
        R1 = -R1
        C1 = -C1

    R2 = U @ W @ Vt
    C2 = -U[:, 2]
    if np.linalg.det(R2) < 0:
        R2 = -R2
        C2 = -C2

    R3 = U @ W.T @ Vt
    C3 = U[:, 2]
    if np.linalg.det(R3) < 0:
        R3 = -R3
        C3 = -C3

    R4 = U @ W.T @ Vt
    C4 = -U[:, 2]
    if np.linalg.det(R4) < 0:
        R4 = -R4
        C4 = -C4

    R_set = np.stack([R1, R2, R3, R4])
    C_set = np.stack([C1, C2, C3, C4])
    return R_set, C_set

def vec2skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

def Triangulation(P1, P2, track1, track2):
    """
    Triangulation method to compute the 3D coordinates of points given their corresponding 2D coordinates in two different camera poses.

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    """
    
    
    Linear Solution
    x = PX
    x is the point projection on image plane
    P is the camera pose
    X is the triangulated point in 3D   
    """
    if track1.shape[0] != track2.shape[0]:
        sys.exit("Your tracks do not match for Triangulation")
    n = track1.shape[0]
    points_3d = np.zeros((n, 3))

    for i in range(n):
        u = np.array([track1[i, 0], track1[i, 1], 1])
        v = np.array([track2[i, 0], track2[i, 1], 1])

        h1 = (vec2skew(u) @ P1)[(0, 1), :]
        h2 = (vec2skew(v) @ P2)[(0, 1), :]
        h3 = np.vstack((h1, h2))
        U, D, Vt = np.linalg.svd(h3)
        # print(Vt) extracts the last row 
        a = Vt[-1] 
        # Homogenize
        points_3d[i] = a[:3] / a[3]
    return points_3d


def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points
    the code checks whether each 3D point is in front of both cameras by comparing their positions with the camera centers and the corresponding 
    viewing directions. The resulting binary vector indicates whether each point satisfies the cheirality condition (1) or not (0).
    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """

    R1, R2 = P1[:, :3], P2[:, :3]
    t1, t2 = P1[:, 3], P2[:, 3]
    # Get back to original position
    C1, C2 = -R1.T @ t1, -R2.T @ t2
    # row vectors 
    r3_1, r3_2 = R1[2, :], R2[2, :]
    # If the dot product is greater than 0, it means the point is in front of the camera.
    cond1 = ((X - C1) @ r3_1) > 0
    cond2 = ((X - C2) @ r3_2) > 0
    # Reshape the resulting boolean array into a 1D binary vector.
    valid_index = np.logical_and(cond1, cond2).reshape(-1)
    return valid_index



def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    N = track1.shape[0]

    # Find correspondences for pair im1 and im2
    mask = np.logical_and(np.sum(track1, axis=1) != -2, np.sum(track2, axis=1) != -2)
    x1, x2 = track1[mask], track2[mask]
    featureIdx = np.asarray(np.nonzero(mask)[0])

    E, inlier = EstimateE_RANSAC(x1, x2, 500, 0.003)

    R_set, C_set = GetCameraPoseFromE(E)

    numValid = 0
    valid3DCoord = []
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    for i in range(4):
        P2 = np.hstack([R_set[i], -(R_set[i] @ C_set[i]).reshape((3, 1))])
        X_curr = Triangulation(P1, P2, x1, x2)
        Index_Cheirality = EvaluateCheirality(P1, P2, X_curr)
        valid3DCoord.append(X_curr[Index_Cheirality])

        if np.sum(Index_Cheirality) > numValid:
            numValid = np.sum(Index_Cheirality)
            R = R_set[i]
            C = C_set[i]
            X = -1 * np.ones((N, 3))
            X[featureIdx[Index_Cheirality]] = X_curr[Index_Cheirality]
    return R, C, X
