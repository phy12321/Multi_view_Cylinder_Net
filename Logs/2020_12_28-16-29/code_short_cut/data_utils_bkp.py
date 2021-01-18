import os
import numpy as np
import logging
import torch
import time
from scipy.spatial.transform import Rotation as R
import math
from .pc_utils import points2cylinder_map


def slp():
    time.sleep(33333)


def invert(Tr):
    """
    :param Tr: numpy.array, Homogeneous transformation matrix, (4,4),(3,4),(1,12)
    :return: numpy.array, invert of Homogeneous transformation matrix
    """
    if Tr.shape not in [(4, 4), (3, 4), (1, 12)]:
        raise ValueError("invalid matrix shape")
    Tr = np.array(Tr, dtype=np.float64)
    if Tr.shape == (1, 12):
        Tr = Tr.reshape((3, 4))
    if Tr.shape == (3, 4):
        Tr = np.concatenate([Tr, np.array([[.0, .0, .0, 1.0]])], axis=0)
    Tr = np.mat(Tr)
    return Tr.I


def matrix_rot_trans_error(pred, label):
    """
    matrix rot error and L2 trans error
    :param pred: list , [rot , trans]
    :param label: list , [rot, trans]
    :return:
    """
    batch_size = pred[0].shape[0]

    rot_pred = np.array(pred[0])
    rot_label = np.array(label[0])

    trans_label = np.array(label[1])
    trans_pred = np.array(pred[1])

    ## trans error:
    trans_error = np.linalg.norm(trans_label - trans_pred, axis=1)

    # compute error in matrix form all the time
    if rot_pred.shape[-1] == 4:  # quaternion
        rot_pred = quar2matrix(rot_pred, size=(3, 3))
        rot_pred = rot_pred.reshape((batch_size, -1))
    elif rot_pred.shape[-1] == 3:  # euler
        rot_pred = euler2matrix(rot_pred, size=(3, 3))
        rot_pred = rot_pred.reshape((batch_size, -1))

    assert rot_pred.shape[-1] == 9
    assert rot_label.shape[-1] == 9
    ## matrix rot error:
    rot_error = matrix_rot_error(rot_pred, rot_label)
    return rot_error, trans_error


def matrix_rot_error(rot_pred, rot_label):
    """
    error of rot matrix between pred and label
    :param rot_pred: (batch_size, 9),tensor
    :param rot_label: [(batch_size, 9),(batch_size, 3)] , rot , trans , tensor
    :return: error of angular and trans : (B,) (B,)
    """
    rot_label = np.array(rot_label.reshape(-1, 9))
    rot_pred = np.array(rot_pred.reshape(-1, 9))

    rot_F_norm = np.linalg.norm(rot_pred - rot_label, axis=1)

    angular_error = []
    for i in range(rot_F_norm.shape[0]):
        angular_error.append(2 * math.asin(rot_F_norm[i] / np.sqrt(8)))
    angular_error = np.array(angular_error)
    return angular_error


def euler2matrix(euler, size=(4, 4)):
    """
    :param euler:  (B,3) (rot,)  or (B, 6) , (rot , trans)
    :param size: size of return matrix
    :return: rot matrix
    """
    rot = R.from_euler('zyx', euler[:3], degrees=True)
    rot = R.as_matrix(rot)
    if size == (4, 4):
        matrix = np.concatenate([np.concatenate([rot, euler[3:, None]], axis=1),
                                 np.array([[.0, .0, .0, 1.0]])], axis=0)
    elif size == (3, 4):
        matrix = np.concatenate([rot, euler[3:, None]], axis=1)
    elif size == (3, 3):
        matrix = rot
    else:
        raise ValueError("error size")
    return matrix


def quar2matrix(quaternion, size=(4, 4)):
    """
    :type size: matrix size
    :param quaternion: tensor or numpy, (7,) , rot + translation
    :return: transformation matrix
    """
    rot = R.from_quat(quaternion[:4])
    rot = R.as_matrix(rot)
    if size == (4, 4):
        matrix = np.concatenate([np.concatenate([rot, quaternion[4:, None]], axis=1),
                                 np.array([[.0, .0, .0, 1.0]])], axis=0)
    elif size == (3, 4):
        matrix = np.concatenate([rot, quaternion[4:, None]], axis=1)
    elif size == (3, 3):
        matrix = rot
    else:
        raise ValueError("error size")
    return matrix


def get_logger(log_path):
    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    log_name = os.path.join(log_path, 'info.log')
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    return logger


def Rel_Pose2Abs_Pose(rel_pose_list, calib):
    """
    transform relative pose into abs pose list, for evo_traj
    :param calib: calib matrix for each sequence, (1,12)
    :param rel_pose_list: (N,12) or (N,7) , list!
    :return: (N+1, 12), in transform matrix format
    """
    calib_T = np.concatenate([calib.reshape((3, 4)), np.array([[.0, .0, .0, 1.0]])], axis=0)
    if rel_pose_list[0].shape[-1] == 7:  # convert Quaternion into matrix
        rel_pose_list = [np.squeeze(np.reshape(quar2matrix(pose, size=(3, 4)), (1, -1))) for pose in rel_pose_list]
    elif rel_pose_list[0].shape[-1] == 6:  # convert Euler into matrix
        rel_pose_list = [np.squeeze(np.reshape(euler2matrix(pose, size=(3, 4)), (1, -1))) for pose in rel_pose_list]

    rel_pose = np.array(rel_pose_list)
    abs_pose = np.zeros((rel_pose.shape[0] + 1, 12))
    abs_pose[0, :] = [1.0, .0, .0, .0, .0, 1.0, .0, .0, .0, .0, 1.0, .0]
    T_cur = np.concatenate([abs_pose[0, :].reshape((3, 4)), np.array([[.0, .0, .0, 1.0]])], axis=0)
    for i in range(rel_pose.shape[0]):
        rel_T = np.concatenate([rel_pose[i, :].reshape((3, 4)), np.array([[.0, .0, .0, 1.0]])], axis=0)
        # print('rel_T: \n',np.round(rel_T, 2))
        T_cur = np.array(np.mat(T_cur) * np.mat(calib_T) * np.mat(rel_T) * invert(calib_T))
        # print('T_cur: \n',np.round(T_cur, 2),'\n\n')
        abs_pose[i + 1, :] = np.reshape(T_cur[:3, :], (1, -1))
        # time.sleep(3)
    return abs_pose


if __name__ == "__main__":
    calib = np.eye(4)[:3, :]
    a = np.eye(4)
    a[:3, -1] = [2.0, .0, .0]
    a = a[:3, :]
    a = a.reshape((1, -1))[0]
    rel_pose = [a, a, a, a, a]

    abs_pose = Rel_Pose2Abs_Pose(rel_pose, calib)
    print(abs_pose)
    slp()
