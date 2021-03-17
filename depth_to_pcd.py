import numpy as np


def depth2pcd(depth):
    def remap(x, in_range_l, in_range_r, out_range_l, out_range_r):
        return (x - in_range_l) / (in_range_r - in_range_l) * (out_range_r - out_range_l) + out_range_l

    # depth = remap(depth, depth.min(), depth.max(), 0, 1)
    # print(depth)
    scalingFactor = 1
    fovy = 60
    aspect = depth.shape[1] / depth.shape[0]
    # fovx = 2 * math.atan(math.tan(fovy * 0.5 * math.pi / 360) * aspect)
    width = depth.shape[1]
    height = depth.shape[0]
    fovx = 2 * math.atan(width * 0.5 / (height * 0.5 / math.tan(fovy * math.pi / 360 / 2))) / math.pi * 360
    fx = width / 2 / (math.tan(fovx * math.pi / 360 / 2))
    fy = height / 2 / (math.tan(fovy * math.pi / 360 / 2))
    points = []

    for v in range(0, height, 10):
        for u in range(0, width, 10):
            Z = depth[v][u] / scalingFactor
            if Z == 0:
                continue
            X = (u - width / 2) * Z / fx
            Y = (v - height / 2) * Z / fy
            points.append([X, Y, Z])

    return np.array(points)