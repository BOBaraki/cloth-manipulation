import numpy as np


def get_camera_extrinsic_matrix(azimuth, distance, elevation, lookat):
    """Get the camera extrinsic matrix converting world coordinates to the camera referential.

    Args:
        azimuth (float): Azimuth angle in radians.
        distance (float): Distance to point to which the camera is looking at.
        elevation (float): Elevation angle in radians.
        lookat (np.ndarray): 3D position (expressed in world coordinates)
            of the point at which the camera is looking at.

    Returns:
        np.ndarray: 3x4 extrinsic matrix.
    """
    rotation_matrix = (
        rotate_z(np.pi) @ rotate_x(elevation + np.pi / 2) @ rotate_z(azimuth)
    )
    translation_vector = np.array([0, 0, distance]) - rotation_matrix @ lookat
    return np.c_[rotation_matrix, translation_vector]


def get_camera_intrinsic_matrix(width, height, vertical_fov):
    """Get the camera intrinsic matrix converting camera coordinates to image coordinates.

    Args:
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        vertical_fov (float): Vertical field of view in radians.

    Returns:
        np.ndarray: 3x3 intrinsic matrix.
    """
    px, py = (width / 2, height / 2)

    fy = height / (2.0 * np.tan(vertical_fov / 2.0))

    horizontal_fov = 2.0 * np.arctan(np.tan(vertical_fov / 2) * width / height)
    fx = width / (2.0 * np.tan(horizontal_fov / 2.0))

    return np.array([[fx, 0, px], [0, fy, py], [0, 0, 1.0]])


def rotate_x(theta):
    """Get the rotation matrix encoding a rotation along the x-axis.

    Args:
        theta (float): Angle of rotation in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def rotate_z(theta):
    """Get the rotation matrix encoding a rotation along the z-axis.

    Args:
        theta (float): Angle of rotation in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
