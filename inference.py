import time
import torch
import torch.nn as nn
import numpy as np
import os
from utils.pc_utils import points2multi_view_range_map, display_points, display_map, points2cylinder_map
from utils.data_utils import quar2matrix, euler2matrix
from time import sleep
import shutil
from utils import kitti_data_set
import argparse
from torch.utils.data import DataLoader
import cv2

project_path = os.getcwd()
log_path = os.path.join(project_path, 'Logs', '2020_12_28-16-29')

ckp_path = os.path.join(log_path, 'checkpoints', 'best.pth')
data_path = os.path.join(project_path, 'dataset', 'kitti')
model_path = os.path.join(log_path, 'code_short_cut', "net_bkp.py")
shutil.copy(model_path, project_path)

args = argparse.ArgumentParser("inference")
args.add_argument('--batch_size', type=int, default=1)
args.add_argument('--data_set', type=str, default='dataset/kitti')
args.add_argument('--num_workers', type=int, default=1)
args.add_argument('--test_seq', type=list, default=['02'])
args.add_argument('--num_sample', type=int, default=None)
args.add_argument('--test_dataset', type=bool, default=False)
args.add_argument('--overlap', type=int, default=20)
args.add_argument('--rot_loss_weight', type=float, default=1)
args.add_argument('--trans_loss_weight', type=float, default=1)
args.add_argument('--save_pose', type=bool, default=True)  # for evo trajectory
arg_list = args.parse_args()


def load_net(debug):
    from net_bkp import Net
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    net = Net(in_channel=2, rot_channel=3)
    net.load_state_dict(ckp['net'])
    net.eval()
    print("load ckp from epoch ", ckp['epoch'])
    os.remove(os.path.join(project_path, "net_bkp.py"))

    return net


class FeatureExtractor(nn.Module):
    def __init__(self, submodule):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, data):
        outputs = {}

        img1, img2, rel_T_12, src_path, tgt_path = data  ## [l,r,h,b][l,r,h,b][rot,trans]
        left1, right1, head1, back1, map1 = img1[0].cuda(), img1[1].cuda(), \
                                            img1[2].cuda(), img1[3].cuda(), img1[4].cuda()
        left2, right2, head2, back2, map2 = img2[0].cuda(), img2[1].cuda(), \
                                            img2[2].cuda(), img2[3].cuda(), img2[4].cuda()

        out_rot, out_transl, lr_flow, lr_out_conv1, lr_out_conv2, lr_out_conv3, lr_out_deconv2, \
        hb_flow, hb_out_conv1, hb_out_conv2, hb_out_conv3, hb_out_deconv2, cy_list = self.submodule(
            [left1, left2], [right1, right2],
            [head1, head2], [back1, back2],
            [map1, map2])  # [B, 7] ,

        # outputs["out_rot"] = out_rot
        # outputs["out_transl"] = out_transl
        outputs["input"] = map1
        outputs["lr_flow"] = lr_flow
        outputs["lr_out_conv1"] = lr_out_conv1
        outputs["lr_out_conv2"] = lr_out_conv2
        outputs["lr_out_conv3"] = lr_out_conv3
        outputs["lr_out_deconv2"] = lr_out_deconv2

        outputs["hb_flow"] = hb_flow
        outputs["hb_out_conv1"] = hb_out_conv1
        outputs["hb_out_conv2"] = hb_out_conv2
        outputs["hb_out_conv3"] = hb_out_conv3
        outputs["hb_out_deconv2"] = hb_out_deconv2
        for i in range(len(cy_list)):
            outputs[f"cy_map_{i}"] = cy_list[i]

        return out_rot, out_transl, outputs


def visiual_feature():
    net = load_net(debug=True)

    print("Inference on the data from ", data_path)
    test_dataset = kitti_data_set.KittiDataset(mode="test", data_path=data_path,
                                               seq_list=arg_list.test_seq, rot_form="euler")
    test_dataloader = DataLoader(test_dataset, batch_size=arg_list.batch_size, shuffle=False,
                                 num_workers=arg_list.num_workers, pin_memory=True, drop_last=True)

    dataloader = iter(test_dataloader)
    for i in range(np.random.randint(len(dataloader))):
        data = next(dataloader)

    my_extractor = FeatureExtractor(net).cuda()
    _, _, outs = my_extractor(data)
    save_path = os.path.join(log_path, 'visualization')
    therd_size = 99999
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            if 'fc' in k:
                continue

            feature = features.data.cpu().numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

            dst_path = os.path.join(save_path, k)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)

            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(feature_img.shape[0]) + '_' + str(
                    feature_img.shape[1]) + '.png')

                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (1016, 72), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)

            # dst_file = os.path.join(dst_path, str(i) + '.png')
            # cv2.imwrite(dst_file, feature_img)


def inference_on_samples():
    net = load_net(debug=False)

    print("Inference on the data from ", data_path)
    test_dataset = kitti_data_set.KittiDataset(mode="test", data_path=data_path,
                                               seq_list=arg_list.test_seq, rot_form="euler")
    test_dataloader = DataLoader(test_dataset, batch_size=arg_list.batch_size, shuffle=False,
                                 num_workers=arg_list.num_workers, pin_memory=True, drop_last=True)

    with torch.no_grad():
        start = time.time()
        net = net.eval().cuda()
        for idx, data in enumerate(test_dataloader):
            print(f"{idx}/{len(test_dataloader)}")
            img1, img2, rel_T_12, src_path, tgt_path = data  ## [l,r,h,b][l,r,h,b][rot,trans]
            left1, right1, head1, back1, map1 = img1[0].cuda(), img1[1].cuda(), \
                                                img1[2].cuda(), img1[3].cuda(), img1[4].cuda()
            left2, right2, head2, back2, map2 = img2[0].cuda(), img2[1].cuda(), \
                                                img2[2].cuda(), img2[3].cuda(), img2[4].cuda()

            rot_gt, trans_gt, rel_matrix_gt = rel_T_12[0].cuda(), rel_T_12[1].cuda(), rel_T_12[2].cuda()

            rot_pred, trans_pred = net([left1, left2], [right1, right2],
                                       [head1, head2], [back1, back2],
                                       [map1, map2])  # [B, 7] ,
            gt = torch.cat([rot_gt, trans_gt], dim=1)
            gt = np.array(gt.cpu())[0]

            out_put = torch.cat([rot_pred, trans_pred], dim=1)
            out_put = np.array(out_put.cpu())[0]

            rot_error = torch.sum(torch.abs(rot_gt - rot_pred)).detach().cpu().numpy()
            trans_error = torch.sum(torch.abs(trans_gt - trans_pred)).detach().cpu().numpy()

            if idx > 133 and (rot_error > .8 and trans_error > 0.7):
                print(f"rot_error: {rot_error}; trans_error: {trans_error}")
                print(f"rot_pred: {rot_pred.detach().cpu().numpy()}; rot_gt: {rot_gt.detach().cpu().numpy()}")
                print(f"trans_pred: {trans_pred.detach().cpu().numpy()}; trans_gt: {trans_gt.detach().cpu().numpy()}")
                tgt_pt = np.fromfile(tgt_path[0], count=-1, dtype=np.float32).reshape([-1, 4])[:, :3]
                src_pt = np.fromfile(src_path[0], count=-1, dtype=np.float32).reshape([-1, 4])[:, :3]
                rel_matrix = euler2matrix(out_put)
                src_pt = np.concatenate([src_pt, np.ones((src_pt.shape[0], 1))], axis=1)
                tgt_from_src = np.matmul(rel_matrix, src_pt.T).T
                tgt_from_src_gt = np.matmul(rel_matrix_gt[0].detach().cpu().numpy(), src_pt.T).T
                tgt_pred = points2cylinder_map(tgt_from_src)
                tgt_gt = points2cylinder_map(tgt_from_src_gt)
                error_map_gt = np.abs(tgt_gt - map1.detach().cpu().numpy()[0, :, :, :])
                error_map = np.abs(tgt_gt - tgt_pred)
                error = np.abs(error_map - error_map_gt)
                display_map(tgt_gt[0, :, :],  # tgt_gt
                            tgt_pred[0, :, :],  # pred_tgt
                            error_map[0, :, :],
                            error_map_gt[0, :, :],
                            error[0, :, :])
                display_points(src_pt, tgt_from_src, tgt_from_src_gt)  # (red, purple, blue)


if __name__ == "__main__":
    inference_on_samples()
    # visiual_feature()
