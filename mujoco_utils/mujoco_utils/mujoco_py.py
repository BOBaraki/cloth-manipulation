import numpy as np
from mujoco_py.cymj import PyMjvCamera
from mujoco_utils.camera import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix


def get_camera_transform_matrix(width, height, vertical_fov, camera: PyMjvCamera):
    """Get the camera matrix from a mujoco_py camera converting world coordinates to image coordinates.

    Args:
        width (int): Width of the rendered image.
        height (int): Height of the rendered image.
        vertical_fov (float): Vertical field of view in radians.
        camera (PyMjvCamera): mujoco_py camera with azimuth, distance, elevation, and lookat attributes.

    Returns:
        np.ndarray: 3x4 camera matrix.
    """
    extrinsic_matrix = get_camera_extrinsic_matrix(
        azimuth=np.radians(camera.azimuth),
        distance=camera.distance,
        elevation=np.radians(camera.elevation),
        lookat=camera.lookat,
    )
    intrinsic_matrix = get_camera_intrinsic_matrix(
        width=width,
        height=height,
        vertical_fov=np.radians(vertical_fov),
    )
    return intrinsic_matrix @ extrinsic_matrix
