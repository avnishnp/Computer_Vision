import numpy as np

def Rotation2Quaternion(R):
    """
    Convert a rotation matrix to quaternion
    
    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix

    Returns
    -------
    q : ndarray of shape (4,)
        The unit quaternion (w, x, y, z)
    """
    # Extract the elements of the rotation matrix
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
    
    # Calculate the trace of the rotation matrix
    trace = r11 + r22 + r33
    
    if trace > 0:
        # Case 1: Trace is positive (w > 0.5)
        s = 0.5 / np.sqrt(1.0 + trace)
        w = 0.25 / s
        x = (r32 - r23) * s
        y = (r13 - r31) * s
        z = (r21 - r12) * s
    else:
        # Cases 2 and 3: Trace is non-positive
        if r11 > r22 and r11 > r33:
            # Case 2: r11 is the largest diagonal term
            s = 2.0 * np.sqrt(1.0 + r11 - r22 - r33)
            w = (r32 - r23) / s
            x = 0.25 * s
            y = (r12 + r21) / s
            z = (r13 + r31) / s
        elif r22 > r33:
            # Case 3: r22 is the largest diagonal term
            s = 2.0 * np.sqrt(1.0 + r22 - r11 - r33)
            w = (r13 - r31) / s
            x = (r12 + r21) / s
            y = 0.25 * s
            z = (r23 + r32) / s
        else:
            # Case 4: r33 is the largest diagonal term
            s = 2.0 * np.sqrt(1.0 + r33 - r11 - r22)
            w = (r21 - r12) / s
            x = (r13 + r31) / s
            y = (r23 + r32) / s
            z = 0.25 * s
    
    # Return the unit quaternion (w, x, y, z)
    q = np.array([w, x, y, z])
    return q


def Quaternion2Rotation(q):
    """
    Convert a quaternion to rotation matrix
    
    Parameters
    ----------
    q : ndarray of shape (4,)
        Unit quaternion (w, x, y, z)

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Calculate the elements of the rotation matrix
    r11 = 1 - 2 * (y**2 + z**2)
    r12 = 2 * (x * y - w * z)
    r13 = 2 * (x * z + w * y)
    r21 = 2 * (x * y + w * z)
    r22 = 1 - 2 * (x**2 + z**2)
    r23 = 2 * (y * z - w * x)
    r31 = 2 * (x * z - w * y)
    r32 = 2 * (y * z + w * x)
    r33 = 1 - 2 * (x**2 + y**2)
    
    # Return the rotation matrix
    R = np.array([[r11, r12, r13],
                  [r21, r22, r23],
                  [r31, r32, r33]])
    return R
