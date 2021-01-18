import torch
import math
import time
# import kornia
import torch.nn.functional as F
from args import arg_list
from utils.msssim_utils import msssim
from utils.pc_utils import display_map

def SLP(t=3333):
    time.sleep(t)

def tensor_total_loss(src_pt, tgt_pt, num_src_pt, num_tgt_pt,
                      rot_pred, trans_pred, rot_gt, trans_gt, rot_matrix_gt, pose_norm):

    rot_matrix_pred = tensor_euler_trans2matrix(torch.cat([rot_pred, trans_pred], dim=1))
    error_map, error_map_gt = tensor_error_map(src_pt, tgt_pt,num_src_pt, num_tgt_pt,
                                               rot_matrix_pred, rot_matrix_gt)
    # pose_loss:  quaternion , euler, flatten matrix
    rot_loss, trans_loss = pose_L2_loss([rot_pred, trans_pred], [rot_gt, trans_gt], norm=pose_norm)
    pose_loss = arg_list.rot_loss_w * rot_loss + arg_list.trans_loss_w * trans_loss
    # reproj_depth_loss:
    reproj_range_loss, error_range, error_range_gt = tensor_reproj_range_loss(error_map[:, 0, :, :],
                                                                              error_map_gt[:, 0, :, :])


    # reproj_normal_loss:
    # reproj_normal_loss, error_normal, error_normal_gt = tensor_reproj_normal_loss(error_map[:, 2:, :, :],
    #                                                                               error_map_gt[:, 2:, :, :])

    return pose_loss, reproj_range_loss, torch.tensor(0)

def pose_L2_loss(pred, label, norm=2):
    """
    rot loss in the form of  Quaternion
    :param pred: (batch_size, 7), tensor   ((i,j,k,w),(x,y,z))
    :param label: [(batch_size, 4),(batch_size, 3)] ,tensor
    :return: loss of Quaternion, tensor
    """
    rot_label = label[0]
    trans_label = label[1]
    rot_pred = pred[0]
    trans_pred = pred[1]
    if norm == 2:
        rot_loss = F.mse_loss(rot_pred, rot_label)
        trans_loss = F.mse_loss(trans_pred, trans_label)
    elif norm == 1:
        rot_loss = F.smooth_l1_loss(rot_pred, rot_label)
        trans_loss = F.smooth_l1_loss(trans_pred, trans_label)

    return rot_loss, trans_loss

def tensor_error_map(src_pt, tgt_pt, num_src_pt, num_tgt_pt, rot_matrix_pred, rot_matrix_gt):
    """
    :param src_pt:  (B,N,6)
    :param tgt_pt: (B,N,6)
    :param rot_matrix_pred: (B,4,4)
    :param rot_matrix_gt: (B,4,4)
    :return:
    """
    batch_size = src_pt.shape[0]
    # project src to tgt_gt:
    pt = torch.cat([src_pt[:, :, :3], torch.ones((batch_size, src_pt.shape[1], 1)).cuda()], dim=2)
    proj_tgt_gt_pt = torch.transpose(torch.matmul(rot_matrix_gt, torch.transpose(pt, 2, 1)), 2, 1)
    if src_pt.shape[-1] == 6:
        proj_tgt_gt_pt = torch.cat([proj_tgt_gt_pt[:, :, :-1], src_pt[:, :, 3:]], dim=-1)
    elif src_pt.shape[-1] == 3:
        proj_tgt_gt_pt = proj_tgt_gt_pt[:,:,:-1]
    proj_tgt_gt_map = tensor_points2cylinder_map(proj_tgt_gt_pt)

    # project src to tgt:
    proj_tgt_pt = torch.transpose(torch.matmul(rot_matrix_pred, torch.transpose(pt, 2, 1)), 2, 1)
    if src_pt.shape[-1] == 6:
        proj_tgt_pt = torch.cat([proj_tgt_pt[:, :, :-1], src_pt[:, :, 3:]], dim=-1)
    elif src_pt.shape[-1] == 3:
        proj_tgt_pt = proj_tgt_pt[:,:,:-1]
    proj_tgt_map = tensor_points2cylinder_map(proj_tgt_pt)

    assert tgt_pt.shape[-1] == src_pt.shape[-1]
    tgt_map = tensor_points2cylinder_map(tgt_pt)

    error_map = torch.abs(proj_tgt_map - tgt_map)
    error_map_gt = torch.abs(proj_tgt_gt_map - tgt_map)

    return error_map, error_map_gt


def tensor_reproj_range_loss(error_range, error_range_gt):
    blur = kornia.filters.MedianBlur((3, 3))

    error_range[torch.where(error_range > 253)] = 0
    error_range = blur(error_range[:, None, :, :].type(torch.float32))[:, 0, :, :]

    error_range_gt[torch.where(error_range_gt > 253)] = 0
    error_range_gt = blur(error_range_gt[:, None, :, :].type(torch.float32))[:, 0, :, :]


    error = torch.mean(error_range)
    error_gt = torch.mean(error_range_gt)
    error_loss = torch.abs(error - error_gt)
    # print(f"error_range: {error.item()}, error_range_gt: {error_gt.item()}, diff: {error_loss.item()}")

    msssim_loss = msssim(error_range[:,None,:,:], error_range_gt[:,None,:,:], normalize="Relu")
    alpha = torch.tensor(arg_list.img_alpha).cuda()
    reproj_depth_loss = (1 - alpha) * error_loss + (alpha * 0.5 * (1 - msssim_loss))
    return reproj_depth_loss, error, error_gt

def tensor_reproj_normal_loss(error_normal,error_normal_gt):
    threshold = 200
    blur = kornia.filters.MedianBlur((3, 3))

    error_normal[torch.where(error_normal < threshold)] = 0
    error_normal = blur(error_normal[:, :, :, :])

    error_normal_gt[torch.where(error_normal_gt < threshold)] = 0
    error_normal_gt = blur(error_normal_gt[:, :, :, :])

    error = torch.mean(error_normal)
    error_gt = torch.mean(error_normal_gt)
    error_loss = torch.abs(torch.mean(error_normal) - torch.mean(error_normal_gt))

    msssim_loss = msssim(error_normal, error_normal_gt, normalize="Relu")
    alpha = torch.tensor(arg_list.img_alpha).cuda()
    geo_normal_loss = (1 - alpha) * error_loss + (alpha * 0.5 * (1 - msssim_loss))

    return geo_normal_loss, error, error_gt

def tensor_euler2rot(euler, degrees=True):
    """
    :param euler: tensor , (B,3), [z,x,y]
    :return: matrix, (B,3,3)
    """
    z, x, y = euler[:,0], euler[:,1], euler[:,2]
    if degrees:
        z, x, y = z * math.pi / 180, \
                  x * math.pi / 180, \
                  y * math.pi / 180

    z = torch.clip(z[:,None,None], -math.pi, math.pi)
    y = torch.clip(y[:,None,None], -math.pi, math.pi)
    x = torch.clip(x[:,None,None], -math.pi, math.pi)

    B = z.shape[0]
    zeros = torch.zeros((B, 1, 1)).cuda()
    ones = torch.ones((B, 1, 1)).cuda()

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    a = torch.cat([cosz, -sinz, zeros],dim=2)
    b = torch.cat([sinz, cosz, zeros],dim=2)
    c = torch.cat([zeros, zeros, ones],dim=2)
    zmat = torch.cat([a, b, c], dim=1)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    a = torch.cat([ones, zeros, zeros],dim=2)
    b = torch.cat([zeros, cosx, -sinx],dim=2)
    c = torch.cat([zeros, sinx, cosx],dim=2)
    xmat = torch.cat([a, b, c], dim=1)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    a = torch.cat([cosy, zeros, siny],dim=2)
    b = torch.cat([zeros, ones, zeros],dim=2)
    c = torch.cat([-siny, zeros, cosy],dim=2)
    ymat = torch.cat([a, b, c], dim=1)

    rot_mat = torch.matmul(torch.matmul(ymat, xmat), zmat)
    return rot_mat

def tensor_euler_trans2matrix(euler_trans):
    """
    convert euler_trans vector into matrix
    :param euler_trans: (B,6)  (rz,rx,ry, x,y,z)
    :return: (B,4,4)
    """
    B = euler_trans.shape[0]
    euler = euler_trans[:,:3]
    trans = euler_trans[:,3:,None]
    rot = tensor_euler2rot(euler)

    vector = torch.cat([torch.zeros((B,1,3)).cuda(),torch.ones((B,1,1)).cuda()],dim=2)
    rot_trans = torch.cat([rot,trans],dim=2)
    matrix = torch.cat([rot_trans,vector],dim=1)
    return matrix

def _scale_to_255(a, min, max, dtype=torch.float32):
    return ((a - min) / float((max - min)) * 255).type(dtype)

def tensor_points2cylinder_map(points, v_res=.42, h_res=0.355, v_fov=(-24.9, 2.0),
                             d_range=(6, 100), y_fudge=2):
    """

    :param points: (B, N, 7) [x,y,z, r , i,j,k], or (B, N,6) [x,y,z, i,j,k] or (B,N,4) [x,y,z,r] or (B,N,3) [x,y,z]
    :param reflection:
    :param normal:
    :param v_res:
    :param h_res:
    :param v_fov:
    :param d_range:
    :param y_fudge:
    :return:
    """
    assert points.shape[-1] in [3, 4, 6, 7]
    B = points.shape[0]
    x = points[:,:, 0]
    y = points[:,:, 1]
    z = points[:,:, 2]
    if points.shape[-1] == 4:
        channel = 2
        r = points[:,:, 3]  # Reflectance
    elif points.shape[-1] == 6:
        channel = 4
        n = points[:,:, 3:]
    elif points.shape[-1] == 7:
        channel = 5
        r = points[:,:, 3]  # Reflectance
        n = points[:,:, 4:]
    else:
        channel = 1
    d = torch.sqrt(x ** 2 + y ** 2)
    
    v_fov_total = -v_fov[0] + v_fov[1]
    v_res_rad = v_res * (math.pi / 180)
    h_res_rad = h_res * (math.pi / 180)

    # project into image coordinates, (B, n)
    x_img = torch.atan2(-y, x) / h_res_rad
    y_img = - (torch.atan2(z, d) / v_res_rad)

    # shift to place where min is (0,0)
    x_min = -360.0 / h_res / 2
    x_img = torch.trunc(-x_img - x_min).type(torch.int64)
    x_max = int(math.ceil(360.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = torch.trunc(y_img - y_min).type(torch.int32)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total / v_res) / (v_fov_total * (math.pi / 180))
    h_below = d_plane * math.tan(-v_fov[0] * (math.pi / 180))
    h_above = d_plane * math.tan(v_fov[1] * (math.pi / 180))
    y_max = int(math.ceil(h_below + h_above + y_fudge))

    # CLIP DISTANCES 
    d_points = torch.clip(d, min=d_range[0], max=d_range[1])
    y_img = torch.clip(y_img, min=y_min, max=y_max).type(torch.int64)

    image = torch.zeros([B, channel, y_max + 1, x_max + 1], dtype=torch.float32).cuda()

    # CONVERT TO IMAGE ARRAY
    cylinder_img = torch.zeros([B, y_max + 1, x_max + 1], dtype=torch.float32).cuda()  # 72 1029
    cylinder_img[:, y_img, x_img] = _scale_to_255(d_points, min=d_range[0], max=d_range[1])
    if channel == 1:
        image[:, 0, :, :] = cylinder_img
    elif channel == 2:
        reflection_img = torch.zeros([B, y_max + 1, x_max + 1], dtype=torch.float32).cuda()  # 72 1029
        reflection_img[:, y_img, x_img] = _scale_to_255(r, min=0, max=1)

        image[:, 0, :, :] = cylinder_img
        image[:, 1, :, :] = reflection_img
    elif channel == 4:
        normal_img = torch.zeros([B, 3, y_max + 1, x_max + 1], dtype=torch.float32).cuda()  # 72 1029
        normal_img[:, 0, y_img, x_img] = _scale_to_255(n[:,:,0], min=0, max=1)
        normal_img[:, 1, y_img, x_img] = _scale_to_255(n[:,:,1], min=0, max=1)
        normal_img[:, 2, y_img, x_img] = _scale_to_255(n[:,:,2], min=0, max=1)
        # normal_img[torch.where(normal_img < 128)] = 0

        image[:, 0, :, :] = cylinder_img
        image[:, 1:, :, :] = normal_img[:,:,:,:]
    elif channel == 5:
        reflection_img = torch.zeros([B, y_max + 1, x_max + 1], dtype=torch.float32).cuda()  # 72 1029
        reflection_img[:, y_img, x_img] = _scale_to_255(r, min=0, max=1)

        normal_img = torch.zeros([B, 3, y_max + 1, x_max + 1], dtype=torch.float32).cuda()  # 72 1029
        normal_img[:, 0, y_img, x_img] = _scale_to_255(n[:,:,0], min=0, max=1)
        normal_img[:, 1, y_img, x_img] = _scale_to_255(n[:,:,1], min=0, max=1)
        normal_img[:, 2, y_img, x_img] = _scale_to_255(n[:,:,2], min=0, max=1)
        # normal_img[torch.where(normal_img < 128)] = 0

        image[:, 0, :, :] = cylinder_img
        image[:, 1, :, :] = reflection_img
        image[:, 2:, :, :] = normal_img[:,:,:,:]

    return image.type(torch.float32)

def tensor_points2multi_view_range_map(points,over_lap=20):
    """

    :param points: point, tensor
    :param over_lap: width pixel to expand based on w/4
    :return: range map and reflection map
    """
    # B,4,h,w
    map = tensor_points2cylinder_map(points)
    W = map.shape[-1]
    w0 = int(math.floor(W/4))

    img_left = map[:,:,:,int((W/2)-(3*w0/2)-over_lap):int((W/2)-(w0/2)+over_lap)]
    img_right = map[:,:,:,int((W/2)+(w0/2)-over_lap):int((W/2)+(3*w0/2)+over_lap)]
    img_head = map[:,:,:,int((W/2)-(w0/2)-over_lap):int((W/2)+(w0/2)+over_lap)]
    img_back = torch.cat([map[:,:,:,int((W/2)+(3*w0/2)-2-over_lap):-2],
                          map[:,:,:,2:int((W/2)-(3*w0/2)+2+over_lap)]],dim=-1)

    return map,img_left,img_right,img_head,img_back

def tensor_abs_gradient(pred):
    """
    :param pred: (B,c,h,w)
    :return: (B,c,2,h,w)
    """
    output = kornia.filters.SpatialGradient()(pred)
    return torch.abs(output[:,:,0,:,:])+torch.abs(output[:,:,1,:,:])



if __name__ == "__main__":
    pass
