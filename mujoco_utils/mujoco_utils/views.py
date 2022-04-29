import numpy as np
from typing import List, Tuple


def get_angles_hemisphere(radius, n_views):
    """Sample points on the surface of a hemisphere and return a collection of azimuth and elevation angle pairs.

    References:
        Algorithm from https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
        Use theta in the range [-np.pi / 2, 0] (hemisphere) and phi in the range [0, 2 * np.pi]

    Args:
        radius (float): Radius of the hemisphere.
        n_views (int): Number of views that the user wants to generate. In practice, the algorithm will take such number
            as an upper bound on the number of generated views (see References).

    Returns:
        List[Tuple[float, float]]: List of tuples of (azimuth, elevation) angles
    """
    n_count = 0
    a = 4 * np.pi * radius / (2 * n_views)
    d = np.sqrt(a)
    m_theta = int(np.round(np.pi / d))
    d_theta = np.pi / m_theta
    d_phi = a / d_theta

    angles = []
    for m in range(m_theta // 2):
        theta = np.pi * (m + 0.5) / m_theta
        m_phi = int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(m_phi):
            phi = 2 * np.pi * n / m_phi
            angles.append((np.degrees(phi), np.degrees(-theta)))
            n_count += 1
    print(f"Created {n_count} / {n_views} views on the hemisphere surface")
    return angles
