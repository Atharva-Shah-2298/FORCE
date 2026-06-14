
import numpy as np

def angle_between(v1, v2):
    """
    Compute the angle between two vectors.

    Parameters
    ----------
    v1 : ndarray (3,)
        First vector.
    v2 : ndarray (3,)
        Second vector.

    Returns
    -------
    angle : float
        Angle between the two vectors.
    """

    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)  # Clip to avoid numerical issues
    return np.arccos(cos_theta)

def is_angle_valid(angle, threshold=30):
    """
    Check if an angle is valid.

    Parameters
    ----------
    angle : float
        Angle to check.
    threshold : float, optional
        Threshold in degrees. Default is 30.

    Returns
    -------
    is_valid : bool
        True if the angle is valid, False otherwise.
    """

    # Convert angle to degrees
    angle_degrees = np.degrees(angle)
    # Check if the angle is not in the restricted ranges
    return not ((-1*threshold <= angle_degrees <= threshold) or (180 - threshold <= angle_degrees <= 180 + threshold))