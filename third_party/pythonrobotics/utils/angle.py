"""Angle utilities required by PythonRobotics Informed RRT*."""

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
    """Create a 2-D rotation matrix from an angle."""
    return Rot.from_euler("z", angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):
    """Apply angle modulo normalization."""
    is_float = isinstance(x, float)
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)
    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi
    if degree:
        mod_angle = np.rad2deg(mod_angle)
    return mod_angle.item() if is_float else mod_angle
