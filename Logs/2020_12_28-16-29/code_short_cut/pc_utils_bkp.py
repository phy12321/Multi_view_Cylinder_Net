import numpy as np
from utils.open3d_utils import display_map, display_points
import os
import cv2
import time
import matplotlib.pyplot as plt


def Bilinear(img):
    """
    :param img: np.array, (c, h ,w)
    :return: 
    """
    img = img.astype(np.uint8)
    BiLinear_interpolation = np.zeros(shape=(img.shape[0], img.shape[1] * 2, img.shape[2] * 2), dtype=np.uint8)
    for i in range(img.shape[0]):
        im = img[i, :, :]
        BiLinear_interpolation[i, :, :] = cv2.resize(src=img[i, :, :], dsize=(im.shape[1] * 2, im.shape[0] * 2),
                                                     interpolation=cv2.INTER_LINEAR)
    return BiLinear_interpolation


def scale_to_255(a, min, max, dtype=np.uint8):
    return ((a - min) / float((max - min)) * 255).astype(dtype)


def points2cylinder_map(points, v_res=0.42, h_res=0.355, v_fov=(-24.9, 2.0),
                        d_range=(6, 100), y_fudge=2):  # v_res: 0.445
    """         0.445, 0.352
            Takes point cloud data as input and creates a 360 degree panoramic image, returned as a numpy array.
            :param points: np.array,(N,C),c>=3
            :param v_res:  vertical angular resolution in degrees. This will influence the height of the output image.
            :param h_res:  horizontal angular resolution in degrees. This will influence the width of the output image.
            :param v_fov:  (tuple of two floats) Field of view in degrees (-min_negative_angle, max_positive_angle)
            :param d_range:(tuple of two floats)   Used for clipping distance values to be within a min and max range.
                            (12,100) is better for photometric loss
            :param y_fudge:A hacky fudge factor to use if the theoretical calculations of vertical image height
                            do not match the actual data. For a Velodyne HDL 64E, set this value to 5.
            :return:  A numpy array, (range map, reflection map, normal map)
            """
    assert points.shape[-1] in [3, 4, 6, 7]
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if points.shape[-1] == 4:
        size = 2
        r = points[:, 3]  # Reflectance
    elif points.shape[-1] == 6:
        size = 4
        n = points[:, 3:]
    elif points.shape[-1] == 7:
        size = 5
        r = points[:, 3]  # Reflectance
        n = points[:, 4:]
    else:
        size = 1

    d = np.sqrt(x ** 2 + y ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # project into image coordinates
    x_img = np.arctan2(-y, x) / h_res_rad
    y_img = - (np.arctan2(z, d) / v_res_rad)

    # shift to place where min is (0,0)
    x_min = -360.0 / h_res / 2
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(360.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total / v_res) / (v_fov_total * (np.pi / 180))
    h_below = d_plane * np.tan(-v_fov[0] * (np.pi / 180))
    h_above = d_plane * np.tan(v_fov[1] * (np.pi / 180))
    y_max = int(np.ceil(h_below + h_above + y_fudge))

    # CLIP DISTANCES
    d_points = np.clip(d, a_min=d_range[0], a_max=d_range[1])
    y_img = np.clip(y_img, a_min=y_min, a_max=y_max).astype(np.int32)

    image = np.zeros([size, y_max + 1, x_max + 1], dtype=np.uint8)
    # CONVERT TO IMAGE ARRAY
    cylinder_img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)  # 72 1029
    cylinder_img[y_img, x_img] = scale_to_255(d_points, min=d_range[0], max=d_range[1])

    if size == 1:
        image[0, :, :] = cylinder_img
    elif size == 2:
        reflection_img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)  # 72 1029
        reflection_img[y_img, x_img] = scale_to_255(r, min=0, max=1)

        image[0, :, :] = cylinder_img
        image[1, :, :] = reflection_img
    elif size == 4:
        normal_img = np.zeros([3, y_max + 1, x_max + 1], dtype=np.uint8)  # 72 1029
        normal_img[0, y_img, x_img] = scale_to_255(n[:, 0], min=0, max=1)
        normal_img[1, y_img, x_img] = scale_to_255(n[:, 1], min=0, max=1)
        normal_img[2, y_img, x_img] = scale_to_255(n[:, 2], min=0, max=1)

        image[0, :, :] = cylinder_img
        image[1, :, :] = normal_img[0, :, :]
        image[2, :, :] = normal_img[1, :, :]
        image[3, :, :] = normal_img[2, :, :]

    elif size == 5:
        reflection_img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)  # 72 1029
        reflection_img[y_img, x_img] = scale_to_255(r, min=0, max=1)

        normal_img = np.zeros([3, y_max + 1, x_max + 1], dtype=np.uint8)  # 72 1029
        normal_img[0, y_img, x_img] = scale_to_255(n[:, 0], min=0, max=1)
        normal_img[1, y_img, x_img] = scale_to_255(n[:, 1], min=0, max=1)
        normal_img[2, y_img, x_img] = scale_to_255(n[:, 2], min=0, max=1)

        image[0, :, :] = cylinder_img
        image[1, :, :] = reflection_img
        image[2, :, :] = normal_img[0, :, :]
        image[3, :, :] = normal_img[1, :, :]
        image[4, :, :] = normal_img[2, :, :]
    return image.astype(np.float32)


def points2multi_view_range_map(points, over_lap=20):
    """
    :param points: point, numpy,(N,4)
    :param over_lap: width pixel to expand based on w/4
    :return: range map and reflection map
    """
    cylinder_img = points2cylinder_map(points)

    W = cylinder_img.shape[-1]
    w0 = int(np.floor(W / 4))

    img_left = cylinder_img[:, :, int((W / 2) - (3 * w0 / 2) - over_lap):int((W / 2) - (w0 / 2) + over_lap)]
    img_right = cylinder_img[:, :, int((W / 2) + (w0 / 2) - over_lap):int((W / 2) + (3 * w0 / 2) + over_lap)]
    img_head = cylinder_img[:, :, int((W / 2) - (w0 / 2) - over_lap):int((W / 2) + (w0 / 2) + over_lap)]
    img_back = np.concatenate([cylinder_img[:, :, int((W / 2) + (3 * w0 / 2) - 2 - over_lap):-2],
                               cylinder_img[:, :, 2:int((W / 2) - (3 * w0 / 2) + 2 + over_lap)]], axis=-1)

    return cylinder_img, img_left, img_right, img_head, img_back


def points2BEV(points,
               res=1,
               side_range=(-30., 30.),  # left-most to right-most
               fwd_range=(-30., 30.),  # back-most to forward-most
               height_range=(-1.4, 3.),  # bottom-most to upper-most
               ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "../dataset/kitti/sequences/04/velodyne")
    points_list = os.listdir(data_path)
    points_list.sort(key=lambda x: int(x[:-4]))

    for index in range(len(points_list)):
        print(index, '/', len(points_list))
        pt1 = np.fromfile(os.path.join(data_path, points_list[index]), dtype=np.float32, count=-1).reshape(-1, 4)
        pt2 = np.fromfile(os.path.join(data_path, points_list[index+4]), dtype=np.float32, count=-1).reshape(-1, 4)

        bev1 = points2BEV(pt1)
        print(bev1.shape, np.min(bev1), np.max(bev1))

        time.sleep(2000)