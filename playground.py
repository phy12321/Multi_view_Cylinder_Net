import numpy as np
import os
from utils.kitti_data_set import KittiDataset
from torch.utils.data import DataLoader
from utils.cupy_utils import CupyCriterion
import torch
from time import sleep
from utils.pc_utils import display_map
from utils.pc_utils import points2cylinder_map, display_map
import time

def dis_error():
    batch_size = 2
    root = os.path.join(os.getcwd(), "dataset", "kitti")
    kitti = KittiDataset(data_path=root, rot_form="euler", seq_list=['04'])
    test_dataloader = DataLoader(kitti, batch_size=batch_size, shuffle=False, num_workers=1)
    criterion = CupyCriterion

    for data in test_dataloader:
        img1, img2, rel_T12, src_path, tgt_path = data  ## (B,N,4),(B,N,4),(B,2,h,w),(B,2,h,w),(B,4,4)
        left1, right1, head1, back1, img1 = img1[0], img1[1], img1[2], img1[3], img1[4]
        left2, right2, head2, back2, img2 = img2[0], img2[1], img2[2], img2[3], img2[4]
        euler = rel_T12[0].detach().cpu().numpy()
        # if np.sum(euler) > 1.8:
        #     continue
        # else:
        #     print(f"np.sum(euler): {np.sum(euler)}; euler: {euler}")

        rot = rel_T12[0]
        trans = rel_T12[1]
        matrix = rel_T12[2]
        error_map = torch.zeros(batch_size, 1, img1.shape[-2], img1.shape[-1]).cuda()
        for i in range(batch_size):
            error = criterion.apply(src_path[0], tgt_path[0], matrix[i, :, :])
            matrix[i, :3, :] = torch.randint(low=3, high=10, size=[3, 4])
            error1 = criterion.apply(src_path[0], tgt_path[0], matrix[i, :, :])

            # error_map[i,0,:,:] = error
            print(f"error sum: {torch.sum(error)}")
            print(f"error1 sum: {torch.sum(error1)}")
            # display_map(
            #     error_map[0,0,:,:],
            #     error_map[1,0,:,:],
            # )
            sleep(33333)

def image_projection_lego_loam(pt):
    ang_res_x,ang_res_y = 0.4, 0.427
    ang_bottom = 24.9
    groundScanInd = 50

    H, W = 64,900
    range_img = np.zeros((H,W))
    for i in range(pt.shape[0]):
        x = pt[i,0]
        y = pt[i,1]
        z = pt[i,2]
        r = np.sqrt(x**2 + y**2)
        if r < 0.1:
            continue
        vertical_ang = np.arctan2(z,r) * 180 / np.pi
        row_idx = H - np.round((vertical_ang + ang_bottom)/ang_res_y)
        if row_idx >= 64 or row_idx < 0:
            continue
        horizon_ang = np.arctan2(x,y) * 180 / np.pi
        column_idx = -np.round((horizon_ang - 90.0)/ang_res_x) + W/2
        if column_idx >= W:
            column_idx -= W
        if column_idx < 0 or column_idx >= W:
            continue
        range_img[int(row_idx),int(column_idx)] = r

    return range_img


def compare_projection_functions(point_path):

    pt = np.fromfile(point_path, dtype=np.float32, count=-1).reshape((-1, 4))[:, :3]

    cy_img = points2cylinder_map(pt)
    cy_img_lego_loam = image_projection_lego_loam(pt)
    display_map(
        cy_img[0,:,:],
        cy_img_lego_loam
    )


if __name__ == "__main__":

    seq_list = ["02"]
    data = os.path.join("D:\\code\\Multi_view_Cylinder_Net\\dataset\\kitti\\sequences",seq_list[0],'velodyne')
    points_list = os.listdir(data)
    points_list.sort(key=lambda x:int(x[:-4]))
    for idx,point_path in enumerate(points_list):
        print(f"{idx}/{len(points_list)},{point_path}")
        point_path = os.path.join(data, point_path)
        compare_projection_functions(point_path)


















