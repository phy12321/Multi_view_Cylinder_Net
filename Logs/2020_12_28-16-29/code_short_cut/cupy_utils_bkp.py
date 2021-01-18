import cupy as cp
import numpy as np
from torch.autograd import Variable, Function
import cupyx.scipy.ndimage
from utils.open3d_utils import display_map
from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import math
from time import sleep
import torch


def cupy_euler2matrix(euler, size=(4, 4)):
    """
    :param euler:  (B,3) (rot,)  or (B, 6) , (rot , trans)
    :param size: size of return matrix
    :return: rot matrix
    """
    rot = cupy_euler2rot(euler[:3], degrees=True)
    if size == (4, 4):
        matrix = cp.concatenate([cp.concatenate([rot, euler[3:, None]], axis=1),
                                 cp.array([[.0, .0, .0, 1.0]])], axis=0)
    elif size == (3, 4):
        matrix = cp.concatenate([rot, euler[3:, None]], axis=1)
    elif size == (3, 3):
        matrix = rot
    else:
        raise ValueError("error size")
    return matrix


def cupy_euler2rot(euler, degrees=True):
    """
    :param euler: cupy , (z,x,y)
    :return: matrix, (3,3)
    """
    if degrees:
        z, x, y = euler[0] * cp.pi / 180, euler[1] * cp.pi / 180, euler[2] * cp.pi / 180
    z = cp.clip(z, -cp.pi, cp.pi)
    y = cp.clip(y, -cp.pi, cp.pi)
    x = cp.clip(x, -cp.pi, cp.pi)

    cosz = cp.cos(z)
    sinz = cp.sin(z)
    a = cp.array([cosz, -sinz, cp.array(0)])
    b = cp.array([sinz, cosz, cp.array(0)])
    c = cp.array([0, 0, 1])
    zmat = cp.concatenate([a[None, :], b[None, :], c[None, :]], axis=0)

    cosx = cp.cos(x)
    sinx = cp.sin(x)
    a = cp.array([1, 0, 0])
    b = cp.array([cp.array(0), cosx, -sinx])
    c = cp.array([cp.array(0), sinx, cosx])
    xmat = cp.concatenate([a[None, :], b[None, :], c[None, :]], axis=0)

    cosy = cp.cos(y)
    siny = cp.sin(y)
    a = cp.array([cosy, cp.array(0), siny])
    b = cp.array([0, 1, 0])
    c = cp.array([-siny, cp.array(0), cosy])
    ymat = cp.concatenate([a[None, :], b[None, :], c[None, :]], axis=0)

    rot_mat = cp.matmul(cp.matmul(ymat, xmat), zmat)
    return rot_mat


def cupy_project_src2tgt(src_pt, rel_T, dtype=cp.uint8):
    assert rel_T.shape == (4, 4)
    normal = src_pt[:, 3:]
    src_pt = cp.concatenate([src_pt[:, :3], cp.ones((src_pt.shape[0], 1))], axis=1)
    tgt_pt_proj = cp.matmul(rel_T, src_pt.T).T
    tgt_pt_proj = cp.concatenate([tgt_pt_proj[:, :3], normal[:, :3]], axis=1)
    proj_tgt_map = cupy_points2cylinder_map(tgt_pt_proj)

    return cp.array(proj_tgt_map, dtype=dtype)


def cupy_points2cylinder_map(points, v_res=.42, h_res=0.355, v_fov=(-24.9, 2.0),
                             d_range=(2, 100), y_fudge=2, clip_255=True, xyz_and_float_coor=False):
    """

    :param xyz_and_float_coor: whether return xyz value and float uvr map.(float uvr: r is not clipped to 0-255)
    :param clip_255: whether to clip the range value to (0,255)
    :param points: (N,7) [x,y,z, r , i,j,k], or (N,6) [x,y,z, i,j,k] or (N,4) [x,y,z,r] or (N,3) [x,y,z]
    :param v_res:
    :param h_res:
    :param v_fov:
    :param d_range:
    :param y_fudge:
    :return:
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

    d = cp.sqrt(x ** 2 + y ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]
    v_res_rad = v_res * (cp.pi / 180)
    h_res_rad = h_res * (cp.pi / 180)

    # project into image coordinates
    x_img = cp.arctan2(-y, x) / h_res_rad
    y_img = - (cp.arctan2(z, d) / v_res_rad)

    # shift to place where min is (0,0)
    x_min = -360.0 / h_res / 2
    if xyz_and_float_coor:
        x_img_f = cp.array(-x_img - x_min, dtype=cp.float32)
    x_img = cp.trunc(-x_img - x_min).astype(cp.int32)
    x_max = int(cp.ceil(360.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    if xyz_and_float_coor:
        y_img_f = cp.array(y_img - y_min, dtype=cp.float32)
    y_img = cp.trunc(y_img - y_min).astype(cp.int32)
    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total / v_res) / (v_fov_total * (cp.pi / 180))
    h_below = d_plane * cp.tan(-v_fov[0] * (cp.pi / 180))
    h_above = d_plane * cp.tan(v_fov[1] * (cp.pi / 180))
    y_max = int(cp.ceil(h_below + h_above + y_fudge))

    # CLIP DISTANCES
    d_points = cp.clip(d, a_min=d_range[0], a_max=d_range[1])
    y_img = cp.clip(y_img, a_min=y_min, a_max=y_max).astype(cp.int32)

    image = cp.zeros([size, y_max + 1, x_max + 1], dtype=cp.float32)

    # CONVERT TO IMAGE ARRAY
    cylinder_img = cp.zeros([y_max + 1, x_max + 1], dtype=cp.float32)  # 72 1029
    if clip_255:
        cylinder_img[y_img, x_img] = _scale_to_255(d_points, min=d_range[0], max=d_range[1])
    else:
        cylinder_img[y_img, x_img] = d_points

    if size == 1:
        image[0, :, :] = cylinder_img
    elif size == 2:
        reflection_img = cp.zeros([y_max + 1, x_max + 1], dtype=cp.float32)  # 72 1029
        reflection_img[y_img, x_img] = _scale_to_255(r, min=0, max=1)

        image[0, :, :] = cylinder_img
        image[1, :, :] = reflection_img
    elif size == 4:
        normal_img = cp.zeros([3, y_max + 1, x_max + 1], dtype=cp.float32)  # 72 1029
        normal_img[0, y_img, x_img] = _scale_to_255(n[:, 0], min=0, max=1)
        normal_img[1, y_img, x_img] = _scale_to_255(n[:, 1], min=0, max=1)
        normal_img[2, y_img, x_img] = _scale_to_255(n[:, 2], min=0, max=1)

        image[0, :, :] = cylinder_img
        image[1:-3, :, :] = normal_img[:, :, :]
    elif size == 5:
        reflection_img = cp.zeros([y_max + 1, x_max + 1], dtype=cp.float32)  # 72 1029
        reflection_img[y_img, x_img] = _scale_to_255(r, min=0, max=1)

        normal_img = cp.zeros([3, y_max + 1, x_max + 1], dtype=cp.float32)  # 72 1029
        normal_img[0, y_img, x_img] = _scale_to_255(n[:, 0], min=0, max=1)
        normal_img[1, y_img, x_img] = _scale_to_255(n[:, 1], min=0, max=1)
        normal_img[2, y_img, x_img] = _scale_to_255(n[:, 2], min=0, max=1)

        image[0, :, :] = cylinder_img
        image[1, :, :] = reflection_img
        image[2:-3, :, :] = normal_img[:, :, :]
    if xyz_and_float_coor:
        uvr = cp.zeros([3, y_max + 1, x_max + 1], dtype=cp.float32)
        uvr[0, y_img, x_img] = cp.clip(y_img_f, a_min=0., a_max=y_max)
        uvr[1, y_img, x_img] = cp.clip(x_img_f, a_min=0., a_max=x_max)
        uvr[2, y_img, x_img] = d_points

        xyz = cp.zeros([3, y_max + 1, x_max + 1], dtype=cp.float32)  # 72 1016
        xyz[0, y_img, x_img] = x
        xyz[1, y_img, x_img] = y
        xyz[2, y_img, x_img] = z
        return image, xyz, uvr
    return image


def cupy_points2multi_view_range_map(points, over_lap=20, dtype="cupy"):
    """

    :param points: point, cupy
    :param over_lap: width pixel to expand based on w/4
    :return: range map and reflection map
    """
    # B,4,h,w
    map = cupy_points2cylinder_map(points)
    W = map.shape[-1]
    w0 = int(math.floor(W / 4))

    img_left = map[:, :, int((W / 2) - (3 * w0 / 2) - over_lap):int((W / 2) - (w0 / 2) + over_lap)]
    img_right = map[:, :, int((W / 2) + (w0 / 2) - over_lap):int((W / 2) + (3 * w0 / 2) + over_lap)]
    img_head = map[:, :, int((W / 2) - (w0 / 2) - over_lap):int((W / 2) + (w0 / 2) + over_lap)]
    img_back = cp.concatenate([map[:, :, int((W / 2) + (3 * w0 / 2) - 2 - over_lap):-2],
                               map[:, :, 2:int((W / 2) - (3 * w0 / 2) + 2 + over_lap)]], axis=-1)
    if dtype == "tensor":
        return torch.tensor(map).cuda(), \
               torch.tensor(img_left).cuda(), \
               torch.tensor(img_right).cuda(), \
               torch.tensor(img_head).cuda(), \
               torch.tensor(img_back).cuda()

    return map, img_left, img_right, img_head, img_back


def _scale_to_255(a, min, max):
    return ((a - min) / float((max - min)) * 255)


def compute_stable_range_map_pixel_mask_(range_map, dist_threshold=4):
    """
    compute the stable surface points of input range map

    :param range_map:
    :param dist_threshold:
    :return: mask
    """
    mask = cp.ones((3, 3))
    for i in range(3):
        for j in range(3):
            if (i + j) % 2 == 0:
                mask[i, j] = 0
    mask[1, 1] = 1

    dist_max = cupyx.scipy.ndimage.maximum_filter(range_map[0, :, :], footprint=mask, mode="wrap")
    dist_min = cupyx.scipy.ndimage.minimum_filter(range_map[0, :, :], footprint=mask, mode="wrap")
    uncer = cp.abs(dist_max - dist_min)
    mask = cp.where(uncer > dist_threshold, 0, 1)
    return mask


def compute_stable_pixel_mask(pt):
    """
    compute the stable surface points of input points
    :param pt:
    :param dist_threshold:  dist(max_neibour - min_neibour) where lower than threshold will be kept.
    :return: range map , xyz map , and stable mask
    """
    dist, xyz, uvr = cupy_points2cylinder_map(pt, clip_255=False, xyz_and_float_coor=True)
    mask = compute_stable_range_map_pixel_mask_(dist)

    return dist, mask, xyz, uvr


def min_max(x):
    print(cp.min(x), ';  ', cp.max(x))


def bilinear_interpolate_range(UVR, UVR_gt):
    """ ilinear_interpolate:
                ----------------------------------------> v
                |           v1               v2
                |                      v
                \  u1      r_11 . . . . . r_21
                |            .               .
                \   u        .      (u,v)    .
                |            .               .
                \            .               .
                |            .               .
                \            .               .
                |  u2      r_12 . . . . . r_22
                u
    :param UVR: tgt_pred_uvr map , [3,h,w]  (float_u,float_v,range), cp.array
    :param UVR_gt: tgt_uvr map ,  [3,h,w]  (float_u,float_v,range),  cp.array
    :return: bilinear appromixed range map, which will be used to sub with range_gt to obtain range error
                [h, w]
    """
    u = UVR[0, :, :]
    v = UVR[1, :, :]
    v1, u1 = cp.meshgrid(cp.arange(UVR.shape[-1]), cp.arange(UVR.shape[-2]))

    delta_u2 = cp.clip(u1 + 1 - u, a_min=0, a_max=1.)
    delta_v2 = cp.clip(v1 + 1 - v, a_min=0, a_max=1.)
    delta_u1 = cp.clip(u - u1, a_min=0, a_max=1.)
    delta_v1 = cp.clip(v - v1, a_min=0, a_max=1.)

    r_11 = UVR_gt[-1, :, :]
    r_12 = cp.pad(UVR_gt[-1, :, :], pad_width=1, mode="wrap")[2:, 1:-1]
    r_22 = cp.pad(UVR_gt[-1, :, :], pad_width=1, mode="wrap")[2:, 2:]
    r_21 = cp.pad(UVR_gt[-1, :, :], pad_width=1, mode="wrap")[1:-1, 2:]

    r = delta_u2 * delta_v2 * r_11 + delta_u2 * delta_v1 * r_21 + \
        delta_u1 * delta_v2 * r_12 + delta_u1 * delta_v1 * r_22
    return r


class CupyPhotoMetricCriterion(Function):
    """
    cupy based loss function:
    """

    @staticmethod
    def forward(ctx, src_path, rel_T, rel_T_gt, tgt_map, error_map_dtype=cp.float32):
        """
        project src_points  to tgt_cylinder_map, and return a error
        between  tgt_cylinder_map and projected_tgt_cylinder_map.
        operate on cupy, return tensor
        """
        src_pt = cp.fromfile(src_path, dtype=cp.float32, count=-1).reshape([-1, 4])
        rot = fromDlpack(to_dlpack(rel_T[0]))
        trans = fromDlpack(to_dlpack(rel_T[1]))
        rot_gt = fromDlpack(to_dlpack(rel_T_gt[0].detach()))
        trans_gt = fromDlpack(to_dlpack(rel_T_gt[1].detach()))
        tgt_map = fromDlpack(to_dlpack(tgt_map.detach()))
        ## TODO: cheack the influence of data type: uint8 and float32: for now using uint8: uint8 is better

        tgt_map = cp.array(tgt_map, dtype=error_map_dtype)
        ## src to tgt , range map and normal map
        rel_T = cupy_euler2matrix(cp.concatenate([rot, trans], axis=0))
        proj_tgt_map = cupy_project_src2tgt(src_pt, rel_T, dtype=error_map_dtype)[0, :, :]
        error = cp.abs(cp.array(tgt_map, dtype=cp.uint8) - cp.array(proj_tgt_map, dtype=cp.uint8))
        error[cp.where(error > 253)] = 0
        error = cupyx.scipy.ndimage.median_filter(error, size=3, output=None,
                                                  mode='constant', cval=0.0, origin=0)

        rel_T_gt = cupy_euler2matrix(cp.concatenate([rot_gt, trans_gt], axis=0))
        proj_tgt_gt = cupy_project_src2tgt(src_pt, rel_T_gt, dtype=error_map_dtype)[0, :, :]
        error_gt = cp.abs(cp.array(tgt_map, dtype=cp.uint8) - cp.array(proj_tgt_gt, dtype=cp.uint8))
        error_gt[cp.where(error_gt > 253)] = 0
        error_gt = cupyx.scipy.ndimage.median_filter(error_gt, size=3, output=None,
                                                     mode='constant', cval=0.0, origin=0)

        return from_dlpack(toDlpack(error)).cuda(), \
               from_dlpack(toDlpack(error_gt)).cuda()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CupyCriterion(Function):
    """
    new criterion of stable correspondence points align
    """

    def forward(ctx, src_path, tgt_path, rel_matrix):
        # TODO: current problem:
        #

        src_pt = cp.fromfile(src_path, dtype=cp.float32, count=-1).reshape([-1, 4])[:, :3]
        tgt_pt = cp.fromfile(tgt_path, dtype=cp.float32, count=-1).reshape([-1, 4])[:, :3]
        rel_matrix = cp.array(rel_matrix)
        # transform src_pt into tgt_pt
        src_pt = cp.concatenate([src_pt[:, :3], cp.ones((src_pt.shape[0], 1))], axis=1)
        tgt_pred_pt = cp.matmul(rel_matrix, src_pt.T).T
        tgt_pred_pt = tgt_pred_pt[:, :3]
        # get cylinder image and stable mask
        tgt_img, tgt_mask, tgt_xyz_img, tgt_uvr = compute_stable_pixel_mask(tgt_pt)
        tgt_pred_img, tgt_pred_mask, tgt_pred_xyz_img, tgt_pred_uvr = compute_stable_pixel_mask(tgt_pred_pt)
        mask = tgt_mask * tgt_pred_mask
        # bilinear interpolate tgt_pred range map from tgt_map
        bilinear_range = bilinear_interpolate_range(tgt_pred_uvr * mask, tgt_uvr * mask)  # [h,w]
        mask = mask * compute_stable_range_map_pixel_mask_(bilinear_range[None, :, :])

        bilinear_range = mask * bilinear_range
        tgt_uvr = mask * tgt_uvr
        error = cp.abs(tgt_uvr[-1, :, :] - bilinear_range)

        return from_dlpack(toDlpack(error)).cuda()


    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
