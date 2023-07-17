import numpy as np
import sys
from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """

    n = x.shape[0]
    # A, which will store the linear equations needed for solving the PnP problem.
    A = []

    for i in range(n):

        # The first four elements [X[i, 0], X[i, 1], X[i, 2], 1] represent the coordinates of the 3D point X[i] in homogeneous form.
        # The next four elements 0, 0, 0, 0 are placeholders.
        # The last four elements -x[i, 0] * X[i, 0], -x[i, 0] * X[i, 1], -x[i, 0] * X[i, 2], -x[i, 0] involve the
        #  multiplication of the x-coordinate of the 2D point x[i] with the corresponding coordinates of the 3D point X[i]

        A.append([X[i, 0], X[i, 1], X[i, 2], 1, 0, 0, 0, 0,
                  -x[i, 0] * X[i, 0], -x[i, 0] * X[i, 1], -x[i, 0] * X[i, 2], -x[i, 0]])

        A.append([0, 0, 0, 0, X[i, 0], X[i, 1], X[i, 2], 1,
                  -x[i, 1] * X[i, 0], -x[i, 1] * X[i, 1], -x[i, 1] * X[i, 2], -x[i, 0]])
    A = np.stack(A)

    # Vh stores the transposed matrix of right singular vectors.
    _, _, Vh = np.linalg.svd(A)

    # The last row of Vh is selected, reshaped to a 3x4 matrix, and assigned to P. This matrix represents the camera projection matrix.
    P = Vh[-1].reshape((3, 4))
    
    # U, D, and Vh represent the left singular vectors, singular values, and transposed right singular vectors, respectively.
    #  "P[:, :3]" would select all rows of the matrix and the first three columns.
    U, D, Vh = np.linalg.svd(P[:, :3])
    R = U @ Vh
    t = P[:, 3] / D[0]

    # This conditional check verifies if the determinant of R is negative, indicating an inconsistent orientation.
    #  In such cases, both R and t are negated to ensure a consistent orientation.
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    # @ operator is used for matrix multiplication. 
    C = -R.T @ t

    return R, C


def PnP_RANSAC(X, x, ransac_n_iter, ransacThreshold):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is 1 if the point is a inlier,
        and 0 otherwise
    """
    n = x.shape[0]
    maxInlier = 0

    # P1 : It represents the initial estimate of the camera projection matrix.
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])

    print('PnP_RANSAC %d pairs' % n)


    # In each iteration, a random sample of 6 3D-2D point correspondences is selected from X and x. 
    # The PnP algorithm is then called to estimate the camera pose (REstimate and CEstimate) using the sampled correspondences.
    for _ in range(ransac_n_iter):
        sampleIdx = np.random.choice(n, 6)
        sampled_X = X[sampleIdx]
        sampled_x = x[sampleIdx]

        REstimate, CEstimate = PnP(sampled_X, sampled_x)


        # The 3D points are transformed to the camera coordinate system (twoD2threeD) using the estimated camera pose. 
        # Then, the 2D points are calculated by normalizing the transformed 3D points. 
        # The Euclidean distance between the original 2D points (x) and the estimated 2D points (xEstimate) is computed and stored in error.
        twoD2threeD = (X - CEstimate) @ REstimate.T
        xEstimate = twoD2threeD[:, :2] / twoD2threeD[:, -1:]
        error = np.linalg.norm(x - xEstimate, axis=1)

        # The number of inliers is determined by counting the number of errors below the ransacThreshold.
        #  If the number of inliers exceeds the current maximum, the maximum is updated, 
        # and the estimated camera pose (REstimate and CEstimate) is stored as the best estimation (R and C).

        numInlier = np.sum(error < ransacThreshold)

        if numInlier > maxInlier:
            maxInlier = numInlier
            R, C = REstimate, CEstimate
            inlier = error < ransacThreshold

    return R, C, inlier


# In summary, this code computes the pose Jacobian by transforming the 3D point to the camera coordinate system,
#  calculating the partial derivatives of the projected 2D point coordinates with respect to the camera pose elements (C and q), 
# and stacking these derivatives into a 2x7 matrix. The resulting matrix represents the pose Jacobian.
def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)

    # The 3D point X is transformed to the camera coordinate system by subtracting the camera center (C) and applying the rotation matrix (R). 
    uvw = R @ (X - C)

    # The derivative of uvw with respect to the camera center (C) is computed and assigned to duvw_dC
    duvw_dC = -R

    duvw_dR = np.zeros((3, 9))
    duvw_dR[0, :3] = X - C
    duvw_dR[1, 3:6] = X - C
    duvw_dR[2, 6:] = X - C

    qw, qx, qy, qz = q
    dR_dq = np.array([[0, 0, -4 * qy, -4 * qz],
                      [-2 * qz, 2 * qy, 2 * qx, -2 * qw],
                      [2 * qy, 2 * qz, 2 * qw, 2 * qx],
                      [2 * qz, 2 * qy, 2 * qx, 2 * qw],
                      [0, -4 * qx, 0, -4 * qz],
                      [-2 * qx, -2 * qw, 2 * qz, 2 * qy],
                      [-2 * qy, 2 * qz, -2 * qw, 2 * qx],
                      [2 * qx, 2 * qw, 2 * qz, 2 * qy],
                      [0, -4 * qx, -4 * qy, 0]])

    duvw_dq = duvw_dR @ dR_dq

    # The derivatives of uvw with respect to C and q are horizontally stacked into a single matrix duvw_dp
    duvw_dp = np.hstack([duvw_dC, duvw_dq])
    if duvw_dp.shape[0] != 3 or duvw_dp.shape[1] != 7:
        sys.exit("Incorrect Jacobian.")
 
    u, v, w = uvw[0], uvw[1], uvw[2]
    du_dp, dv_dp, dw_dp = duvw_dp[0], duvw_dp[1], duvw_dp[2]

    # The pose Jacobian dfdp is computed using the chain rule. The Jacobian elements are calculated based on the partial derivatives
    #  of the projected 2D point coordinates (u and v) with respect to the camera pose elements (C and q)

    dfdp = np.stack([(w * du_dp - u * dw_dp) / w ** 2, (w * dv_dp - v * dw_dp) / w ** 2])
    if dfdp.shape[0] != 2 or dfdp.shape[1] != 7:
        sys.exit("Incorrect Jacobian.")

    return dfdp


def ComputePnPError(R, C, X, b):
    """
    Compute nonlinear PnP estimation error and 1D vector f
    """
    f = (X - C) @ R.T
    f = f[:, :2] / f[:, -1:]
    error = np.average(np.linalg.norm(f - b.reshape(-1, 2), axis=1))
    return error, f.reshape(-1)

# In summary, this code implements an iterative optimization algorithm that refines the camera pose based on the given 3D points and their 2D projections. 
# It aims to minimize the error between the projected 3D points and the actual 2D points in the image. 
# The optimization process iteratively updates the camera pose until a convergence criterion is met or the maximum number of iterations is reached.
def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    RRefined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    CRefined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    maxIter = 50
    eps = 1e-3
    dampingLambda = 1
    n = x.shape[0]
    b = x.reshape(-1)

    previousError, f = ComputePnPError(R, C, X, b)

    for iter in range(maxIter):
        print(' %dth Nonlinear PnP iteration' % iter)
        p = np.concatenate([C, Rotation2Quaternion(R)])

        # It iterates over each 3D point X[i] and computes the pose Jacobian using the ComputePoseJacobian function.
        #  The resulting Jacobians are stored in a list dfdp. Then, the list is vertically stacked to create a matrix of shape 2n x 7.
        dfdp = []
        for i in range(n):
            dfdp.append(ComputePoseJacobian(p, X[i]))
        dfdp = np.vstack(dfdp)  # 2n x 7

        # If  Jacobian matrix dfdp doesn't have the expected shape of 2n x 7, it exits the program with an error message
        if dfdp.shape[0] != 2 * n or dfdp.shape[1] != 7:
            sys.exit("Incorrect Jacobian.")

        # perform the nonlinear optimization steps. 
        dp = np.linalg.inv(dfdp.T @ dfdp + dampingLambda * np.eye(7)) @ dfdp.T @ (b - f)
        C += dp[:3]
        q = p[3:] + dp[3:]
        q = q / np.linalg.norm(q)
        R = Quaternion2Rotation(q)

        error, f = ComputePnPError(R, C, X, b)
        RRefined, CRefined = R, C

        if previousError - error < eps:
            break
        else:
            previousError = error

    return RRefined, CRefined
