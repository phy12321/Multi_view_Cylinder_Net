import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from utils.pc_utils import points2cylinder_map, display_map
from utils.data_utils import euler2matrix
import math
import time
import cv2
import cupy as cp

class Criterion(nn.Module):
    def __init__(self, pose_loss, photometric_loss, geo_loss, sx=-3, sq=-3):
        super(Criterion, self).__init__()
        # self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=True)
        # self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=True)
        self.loss_func = nn.MSELoss()  # F.mse_loss()

def euler2mat(euler):
    """
    :param euler: tensor , (z,x,y)
    :return: matrix, (3,3)
    """
    z, x, y = euler[0] * np.pi / 180, euler[1] * np.pi / 180, euler[2] * np.pi / 180
    z = torch.clip(z, -np.pi, np.pi)
    y = torch.clip(y, -np.pi, np.pi)
    x = torch.clip(x, -np.pi, np.pi)

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zmat = torch.tensor([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.tensor([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.tensor([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])

    rot_mat = torch.matmul(torch.matmul(ymat, xmat), zmat).cuda()
    if euler.shape[-1] == 6:
        transl = euler[3:]
        rot_mat = torch.cat([torch.cat([rot_mat, transl[:,None]],dim=1),torch.tensor([[.0,.0,.0,1.0]]).cuda()],dim=0)
    return rot_mat

class PhotoMetricCriterion(Function):
    @staticmethod
    def forward(ctx,src_points_path, rel_T, rel_T_gt, tgt):
        """
        project src_points  to tgt_cylinder_map, and return a error
        between  tgt_cylinder_map and projected_tgt_cylinder_map.
        operate on numpy, return tensor
        """
        src_pt = np.fromfile(src_points_path, dtype=np.float32, count=-1).reshape([-1, 4])
        rot = rel_T[0].detach().cpu().numpy()
        trans = rel_T[1].detach().cpu().numpy()
        rot_gt = rel_T_gt[0].detach().cpu().numpy()
        trans_gt = rel_T_gt[1].detach().cpu().numpy()
        tgt = tgt.detach().cpu().numpy()
        tgt = np.array(tgt, dtype=np.uint8)

        rel_T = euler2matrix(np.concatenate([rot,trans],axis=0))
        proj_tgt = project_src2tgt(src_pt, rel_T)[0, :, :]
        error = np.abs(proj_tgt - tgt)
        error[np.where(error > 253)] = 0
        error = cv2.medianBlur(error, 3)

        rel_T_gt = euler2matrix(np.concatenate([rot_gt,trans_gt],axis=0))
        proj_tgt_gt = project_src2tgt(src_pt, rel_T_gt)[0, :, :]
        error_gt = np.abs(proj_tgt_gt - tgt)
        error_gt[np.where(error_gt > 253)] = 0
        error_gt = cv2.medianBlur(error_gt, 3)

        return torch.tensor(error).cuda(), torch.tensor(error_gt).cuda()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def project_src2tgt(src_pt, rel_T, normal=False):
    """
    project src points to tgt points, and return a cylinder map of proj_tgt_point
    :type normal:
    :param src_pt:
    :param rel_T:
    :return:
    """
    assert rel_T.shape == (4, 4)
    reflect = src_pt[:, 3]
    src_pt = np.concatenate([src_pt[:, :3], np.ones((src_pt.shape[0], 1))], axis=1)
    tgt_pt_proj = np.matmul(rel_T, src_pt.T).T
    tgt_pt_proj = np.concatenate([tgt_pt_proj[:, :3], reflect[:, None]], axis=1)
    proj_tgt_map = points2cylinder_map(tgt_pt_proj, normal=normal)
    return proj_tgt_map

if __name__ == "__main__":
    rr = [10.0, 50.0, 30.0,8,8,8]

    rot = np.array(euler2mat(torch.tensor(rr)))
    print(rot)
    r = R.from_matrix(rot[:3,:3])
    print(r.as_euler('zxy', degrees=True),'\n')

    r = R.from_euler('zxy', np.array(rr[:3]), degrees=True)
    print(R.as_matrix(r))
    print(r.as_euler('zxy', degrees=True),'\n')
