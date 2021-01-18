import os
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from utils.pc_utils import points2multi_view_range_map, points2BEV, display_points
from scipy.spatial.transform import Rotation as R
import random


def slp(t=3333):
    time.sleep(t)


def invert(Tr):
    """
    :param Tr: numpy.array, Homogeneous transformation matrix
    :return: numpy.array, invert of Homogeneous transformation matrix
    """
    assert Tr.shape == (4, 4)
    Tr = np.mat(Tr)
    return Tr.I


class KittiDataset(Dataset):
    """
    calib.txt:
            Calibration dataset for the cameras: P0/P1 are the 3x4 projection
        matrices after rectification. Here P0 denotes the left and P1 denotes the
        right camera. Tr transforms a point from velodyne coordinates into the
        left rectified camera coordinate system. In order to map a point X from the
        velodyne scanner to a point x in the i'th image plane, you thus have to
        transform it like:
              x = Pi * Tr * X

    Folder 'poses':
            The folder 'poses' contains the ground truth poses (trajectory) for the
        first 11 sequences. This information can be used for training/tuning your
        method. Each file xx.txt contains a N x 12 table, where N is the number of
        frames of this sequence.
            Row i represents the i'th pose of the left camera
        coordinate system (i.e., z pointing forwards) via a 3x4 transformation
        matrix.
            The matrices are stored in row aligned order (the first entries
        correspond to the first row), and take a point in the i'th coordinate
        system and project it into the first (=0th) coordinate system. Hence, the
        translational part (3x1 vector of column 4) corresponds to the pose of the
        left camera coordinate system in the i'th frame with respect to the first
        (=0th) frame. Your submission results must be provided using the same dataset
        format.
    return :
        2 * range map and their relative pose
    """

    def __init__(self, data_path, mode="train", seq_list=None, test_num=None,
                 rot_form='quaternion', train_num_per_class=(4000, 1000, 1000)):
        """
        :param data_path: str
        :param seq_list:
        :param over_lap:
        :param sample_num:
        :param require_points:
        :param transform_representation:
        :param traj_order: step=1 , in time order
        """
        super(KittiDataset, self).__init__()
        self.data_path = data_path
        np.set_printoptions(precision=16)
        self.rot_form = rot_form
        assert mode in ["test", "train"]

        if seq_list is None:
            if mode == "train":
                seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
            elif mode == "test":
                seq_list = ['09', '10']

        rot_scale = [.5, 1.1]

        self.pose_list = os.listdir(os.path.join(data_path, "poses"))
        self.pose_pairs = []  ## (N,5) , [seq_key, pose1_idx,pose2_idx,rel]
        self.pose_pairs1 = []  ## (N,5) , [seq_key, pose1_idx,pose2_idx,rel]
        self.pose_pairs2 = []  ## (N,5) , [seq_key, pose1_idx,pose2_idx,rel]
        self.pose_pairs3 = []  ## (N,5) , [seq_key, pose1_idx,pose2_idx,rel]
        self.points_path_list = {}  ## ([len_seq])
        self.Tr = {}
        ## prepare points dataset path for each sequence
        for seq_name in seq_list:
            seq_path = os.path.join(data_path, 'sequences', seq_name, 'velodyne')
            point_list = os.listdir(seq_path)
            point_list.sort(key=lambda x: int(x[:-4]))
            points_path_list = []
            for i in range(len(point_list)):
                pt_path = os.path.join(seq_path, point_list[i])
                points_path_list.append(pt_path)
            self.points_path_list[str(seq_name)] = points_path_list

        ## pose pairs
        if mode == "test":  # step = 1, time order
            print(f"processing sequence : ", end="")
            for seq_name in seq_list:
                print(f"{seq_name}, ", end="")
                ## trans pose from left camera coordinate system to velodyne coordinate system
                calib = self.__load_calib(os.path.join(data_path, 'sequences', seq_name, 'calib.txt'))
                self.Tr[seq_name] = np.array(calib, dtype=np.float64)
                pose = np.loadtxt(os.path.join(data_path, 'poses', seq_name + '.txt'), dtype=np.float32)
                step = 1
                # step = 1, in time order
                for i in range(pose.shape[0] - step):
                    p1 = pose[i, :]
                    p2 = pose[i + step, :]
                    rel_T_12 = self.__get_rel_transform(p1, p2, calib)  # (4,4)
                    self.pose_pairs.append([str(seq_name), i, i + step, rel_T_12])

            if test_num is not None:
                self.pose_pairs = self.pose_pairs[:test_num]
            print(f"test dataset: {len(self.pose_pairs)}, time order.")

        elif mode == "train":  # step = random int,  in shuffle order
            print(f"processing sequence : ", end="")
            for seq_name in seq_list:
                print(f"{seq_name}, ", end="")
                ## trans pose from left camera coordinate system to velodyne coordinate system
                calib = self.__load_calib(os.path.join(data_path, 'sequences', seq_name, 'calib.txt'))
                self.Tr[seq_name] = np.array(calib, dtype=np.float64)
                pose = np.loadtxt(os.path.join(data_path, 'poses', seq_name + '.txt'), dtype=np.float32)

                for i in range(pose.shape[0] - 3):
                    step = np.random.randint(2, 3)
                    p1 = pose[i, :]
                    p2 = pose[i + 1, :]
                    p3 = pose[i + step, :]
                    rel_T_12 = self.__get_rel_transform(p1, p2, calib)  # (1,16)
                    rel_T_13 = self.__get_rel_transform(p1, p3, calib)

                    if self.__is_in_rot_scale(rel_T_12, [.0, rot_scale[0]]):
                        self.pose_pairs1.append([str(seq_name), i, i + 1, rel_T_12])
                    elif self.__is_in_rot_scale(rel_T_12, [rot_scale[0], rot_scale[1]]):
                        self.pose_pairs2.append([str(seq_name), i, i + 1, rel_T_12])
                    elif self.__is_in_rot_scale(rel_T_12, [rot_scale[1], 10]):
                        self.pose_pairs3.append([str(seq_name), i, i + 1, rel_T_12])

                    if self.__is_in_rot_scale(rel_T_13, [.0, rot_scale[0]]):
                        self.pose_pairs1.append([str(seq_name), i, i + step, rel_T_13])
                    elif self.__is_in_rot_scale(rel_T_13, [rot_scale[0], rot_scale[1]]):
                        self.pose_pairs2.append([str(seq_name), i, i + step, rel_T_13])
                    elif self.__is_in_rot_scale(rel_T_13, [rot_scale[1], 10]):
                        self.pose_pairs3.append([str(seq_name), i, i + step, rel_T_13])

            random.shuffle(self.pose_pairs1)
            random.shuffle(self.pose_pairs2)
            random.shuffle(self.pose_pairs3)
            self.pose_pairs = self.pose_pairs1[:train_num_per_class[0]] + \
                              self.pose_pairs2[:train_num_per_class[1]] + \
                              self.pose_pairs3[:train_num_per_class[2]]
            random.shuffle(self.pose_pairs)

            print(f"train dataset: {len(self.pose_pairs)} / "
                  f"({len(self.pose_pairs1)},{len(self.pose_pairs2)},{len(self.pose_pairs3)}),"
                  f" shuffle, in proportion {train_num_per_class}")
        # self.upsampler = PUGAN(ckp_name='/home/phy12321/code/PU-GAN/log/20201212-1633')

    def __getitem__(self, index):
        """
        src：pt2
        tgt: pt1
        :param index:
        :return:
        """
        seq_name, i, j, rel_T12 = self.pose_pairs[index]
        rel_T12 = rel_T12.reshape((4, 4))
        tgt_path = self.points_path_list[seq_name][i]
        src_path = self.points_path_list[seq_name][j]
        # normal_src_path = os.path.join(self.data_path, 'sequences', seq_name, 'normal', src_path.split('/')[-1])
        # normal_tgt_path = os.path.join(self.data_path, 'sequences', seq_name, 'normal', tgt_path.split('/')[-1])
        pt_tgt_origin = np.fromfile(tgt_path, dtype=np.float32, count=-1)
        pt_src_origin = np.fromfile(src_path, dtype=np.float32, count=-1)
        # src_normal = np.fromfile(normal_src_path, dtype=np.float32, count=-1).reshape([-1, 3])
        # tgt_normal = np.fromfile(normal_tgt_path, dtype=np.float32, count=-1).reshape([-1, 3])

        if pt_tgt_origin.shape[0] % 4 != 0:
            num = pt_tgt_origin.shape[0] // 4
            pt_tgt_origin = pt_tgt_origin[:num * 4].reshape([-1, 4])
            print(f"broken file: {tgt_path}")
        else:
            pt_tgt_origin = pt_tgt_origin.reshape([-1, 4])

        if pt_src_origin.shape[0] % 4 != 0:
            num = pt_src_origin.shape[0] // 4
            pt_src_origin = pt_src_origin[:num * 4].reshape([-1, 4])
            print(f"broken file: {src_path}")
        else:
            pt_src_origin = pt_src_origin.reshape([-1, 4])

        assert pt_tgt_origin.shape[-1] == 4
        assert pt_src_origin.shape[-1] == 4

        if pt_src_origin.shape[0] > 130000:
            pt_src_origin = pt_src_origin[:130000, :]
            # src_normal = src_normal[:130000,:]

        if pt_tgt_origin.shape[0] > 130000:
            pt_tgt_origin = pt_tgt_origin[:130000, :]
            # tgt_normal = tgt_normal[:130000,:]

        num_src_pt = pt_src_origin.shape[0]
        src_pt = np.zeros((130000, 4))
        src_pt[:num_src_pt, :4] = pt_src_origin[:num_src_pt, :4]
        # src_pt[:num_src_pt, 4:] = src_normal[:num_src_pt,:]
        num_tgt_pt = pt_tgt_origin.shape[0]
        tgt_pt = np.zeros((130000, 4))
        tgt_pt[:num_tgt_pt, :4] = pt_tgt_origin[:num_tgt_pt, :4]
        # tgt_pt[:num_tgt_pt, 4:] = tgt_normal[:num_tgt_pt,:]

        img1, left1, right1, head1, back1 = points2multi_view_range_map(pt_tgt_origin,
                                                                        over_lap=20)  # ,projector=self.upsampler)
        img2, left2, right2, head2, back2 = points2multi_view_range_map(pt_src_origin, over_lap=20)  # self.upsampler)

        rot_matrix = rel_T12[:, :].astype(np.float32)
        if self.rot_form == 'quaternion':
            rot_12 = R.from_matrix(rel_T12[:3, :3])
            rot_12 = rot_12.as_quat()
            trans_12 = rel_T12[:3, 3]

        elif self.rot_form == "matrix":
            rot_12 = np.squeeze(rel_T12[:3, :3].reshape((1, -1)))
            trans_12 = rel_T12[:3, 3]
        elif self.rot_form == "euler":
            rot_12 = R.from_matrix(rel_T12[:3, :3])
            rot_12 = rot_12.as_euler('zxy', degrees=True)
            trans_12 = rel_T12[:3, 3]
        else:
            raise ValueError("transform_representation is only in matrix , quaternion or euler")

        rel_T12 = [rot_12.astype(np.float32), trans_12.astype(np.float32), rot_matrix.astype(np.float32)]
        return [[left1, right1, head1, back1, img1], [left2, right2, head2, back2, img2], rel_T12, src_path, tgt_path]

    def __len__(self):
        return len(self.pose_pairs)

    def __get_rel_transform(self, pose1, pose2, calib):
        abs_T1_camera = np.concatenate([pose1.reshape((3, 4)), np.array([[.0, .0, .0, 1.0]])], axis=0)
        abs_T2_camera = np.concatenate([pose2.reshape((3, 4)), np.array([[.0, .0, .0, 1.0]])], axis=0)
        rel_T12_lidar = np.array(invert(calib) * invert(abs_T1_camera) * np.mat(abs_T2_camera) * np.mat(calib))

        return rel_T12_lidar.reshape((1, -1))  # (1,16)

    def __is_in_rot_scale(self, rel_T12, threshold_scale):
        rel_T12 = rel_T12.reshape((4, 4))
        rot_12 = R.from_matrix(rel_T12[:3, :3])
        rot_12 = rot_12.as_euler('zxy', degrees=True)
        abs_rot = abs(rot_12[0]) + abs(rot_12[1]) + abs(rot_12[2])
        return threshold_scale[0] < abs_rot <= threshold_scale[1]

    def __load_calib(self, path):
        calib = np.loadtxt(path, dtype='str')[-1][1:]
        calib = np.array(calib, dtype=np.float64)
        calib = np.concatenate([calib.reshape((3, 4)), np.array([[.0, .0, .0, 1.0]])], axis=0)
        return calib


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.getcwd()), "dataset", "kitti")
    from utils.open3d_utils import display_bev

    kitti = KittiDataset(data_path=root, rot_form="euler", seq_list=['02'])
    test_dataloader = DataLoader(kitti, batch_size=1, shuffle=False, num_workers=1)
    for data in test_dataloader:
        img1, img2, rel_T12, src_path, tgt_path = data  ## (B,N,4),(B,N,4),(B,2,h,w),(B,2,h,w),(B,4,4)
        left1, right1, head1, back1 = img1[0], img1[1], img1[2], img1[3]
        left2, right2, head2, back2 = img2[0], img2[1], img2[2], img2[3]
        euler = rel_T12[0].detach().cpu().numpy()
        if np.sum(euler) > 1.8:
            continue
        else:
            print(np.sum(euler), euler)
        print(src_path)
        print(tgt_path)
        pt_tgt = np.fromfile(tgt_path[0], dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        pt_src = np.fromfile(src_path[0], dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        bev_tgt = points2BEV(pt_tgt)
        bev_src = points2BEV(pt_src)

        rel_T = rel_T12[-1].detach().cpu().numpy()[0, :, :]
        src = np.concatenate([pt_src, np.ones((pt_src.shape[0], 1))], axis=1)
        tgt_pred = np.matmul(rel_T, src.T).T
        print(rel_T.shape, src.shape, tgt_pred.shape)

        bev_tgt_pred = points2BEV(tgt_pred)

        error = np.abs(bev_tgt.astype(np.float32) - bev_src.astype(np.float32))
        error_pred = np.abs(bev_tgt.astype(np.float32) - bev_tgt_pred.astype(np.float32))

        num_pix = bev_tgt.shape[0] * bev_tgt.shape[1]
        print(np.sum(error) / num_pix, np.sum(error_pred) / num_pix)
        display_bev([bev_tgt, bev_tgt_pred, error_pred])
        # display_points(tgt_pred, pt_tgt)
        time.sleep(1)
