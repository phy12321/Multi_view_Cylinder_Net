import torch.nn as nn
import time
import os
from utils.kitti_data_set import KittiDataset
from torch.utils.data import DataLoader
import torch
from model.model_zoo import *


class Net(nn.Module):
    def __init__(self, in_channel, rot_channel, n_base_channels=16, feature_channel=1024, dropout_rate=0.2):
        super(Net, self).__init__()

        self.cylinder_backbone = nn.ModuleList([  # (10, 5, 72, 1016) -> ([10, 512, 4, 18])
            VggBlock(in_channels=in_channel * 2, out_channels=n_base_channels, kernel_size=7, padding=3,
                     maxpool=False),  # 16 torch.Size([10, 16, 128, 416])
            VggBlock(in_channels=n_base_channels, out_channels=n_base_channels, kernel_size=7, padding=3,
                     maxpool=True),  # 16 torch.Size([10, 16, 64, 208])
            VggBlock(in_channels=n_base_channels, out_channels=n_base_channels * 2, kernel_size=5, padding=2,
                     maxpool=False),  # 32 torch.Size([10, 32, 64, 208])
            VggBlock(in_channels=n_base_channels * 2, out_channels=n_base_channels * 2, kernel_size=5, padding=2,
                     maxpool=True),  # 32 torch.Size([10, 32, 32, 104])
            VggBlock(in_channels=n_base_channels * 2, out_channels=n_base_channels * 4, kernel_size=3, padding=1,
                     maxpool=False),  # 64 torch.Size([10, 64, 32, 104])
            VggBlock(in_channels=n_base_channels * 4, out_channels=n_base_channels * 4, kernel_size=3, padding=1,
                     maxpool=False),  # 64 torch.Size([10, 64, 16, 52])
            VggBlock(in_channels=n_base_channels * 4, out_channels=n_base_channels * 8, kernel_size=3, padding=1,
                     maxpool=False),  # 128 torch.Size([10, 128, 16, 52])
            VggBlock(in_channels=n_base_channels * 8, out_channels=n_base_channels * 8, kernel_size=3, padding=1,
                     maxpool=True),  # 128 torch.Size([10, 128, 8, 26])
            VggBlock(in_channels=n_base_channels * 8, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=False),  # 256 torch.Size([10, 256, 8, 26])
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=False),  # 256 torch.Size([10, 256, 4, 13])
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=False),  # 256 torch.Size([10, 256, 4, 13])
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 16, kernel_size=3, padding=1,
                     maxpool=False),  # 256 torch.Size([10, 256, 2, 6])
            VggBlock(in_channels=n_base_channels * 16, out_channels=n_base_channels * 32, kernel_size=3, padding=1,
                     maxpool=False),  # 512 torch.Size([10, 512, 9, 36])
            VggBlock(in_channels=n_base_channels * 32, out_channels=n_base_channels * 32, kernel_size=3, padding=1,
                     maxpool=True),  # 512 torch.Size([10, 512, 4, 18])
        ])

        self.flowNet_backbone = FlowNetS(input_channels=in_channel * 4)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()

        self.rot_fc = nn.Sequential(*[nn.Linear(25088, 512),  # (44100,512),  (25088, 512)
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(512, 128),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(128, 64),
                                      nn.Dropout(0.1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(64, 16),
                                      nn.Dropout(0.1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(16, rot_channel)])

        self.transl_fc = nn.Sequential(*[nn.Linear(25088, 512),  # (44100,512),  (25088, 512)
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(512, 128),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(128, 64),
                                      nn.Dropout(0.1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(64, 16),
                                      nn.Dropout(0.1),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(16, 3)])

    def forward(self,  left_list, right_list, head_list, back_list, map_list):
        cy_map = torch.cat([map_list[0], map_list[1]], dim=1)
        i = 0
        for block in self.cylinder_backbone:
            cy_map = block(cy_map)
            print(i, ": ", cy_map.shape)
            i += 1
        time.sleep(3333)
        cy_map = self.avgpool(cy_map)
        cy_out = self.flatten(cy_map)

        # lr_flow = self.flowNet_backbone(torch.cat([left_list[0], left_list[1],
        #                                            right_list[0], right_list[1]], dim=1))
        # hb_flow = self.flowNet_backbone(torch.cat([head_list[0], head_list[1],
        #                                            back_list[0], back_list[1]], dim=1))
        # lr_flow = self.avgpool(lr_flow)
        # hb_flow = self.avgpool(hb_flow)
        # lr_flow = self.flatten(lr_flow)
        # hb_flow = self.flatten(hb_flow)
        # out = torch.cat([cy_out,lr_flow,hb_flow],dim=-1)

        out = cy_out
        out_rot = 0.01 * self.rot_fc(out)

        out_transl = 0.01 * self.transl_fc(out)

        # out_rot = 0.01 * self.rot3(torch.relu(self.rot2(torch.relu(self.rot1(out)))))
        # out_transl = 0.01 * self.transl3(torch.relu(self.transl2(torch.relu(self.transl1(out)))))
        return out_rot, out_transl


if __name__ == "__main__":
    net = Net(in_channel=2, rot_channel=3)
    root = os.path.join(os.getcwd(), '../dataset/kitti')
    test_dataset = KittiDataset(root, mode="train", seq_list=['04'],
                                                train_num_per_class=[10,10,10],
                                                rot_form="euler")
    test_dataloader = DataLoader(test_dataset, batch_size=10,
                                 shuffle=True, num_workers=2, pin_memory=True)
    net = net.cuda()
    for data in test_dataloader:
        img1, img2, rel_T_12,pt1_path, pt2_path = data  ## [l,r,h,b][l,r,h,b][rot,trans]

        left1, right1, head1, back1, map1 = img1[0].cuda(), img1[1].cuda(), \
                                            img1[2].cuda(), img1[3].cuda(), img1[4].cuda()
        left2, right2, head2, back2, map2 = img2[0].cuda(), img2[1].cuda(), \
                                            img2[2].cuda(), img2[3].cuda(), img2[4].cuda()
        rot_12, trans_12 = rel_T_12[0].cuda(), rel_T_12[1].cuda()
        batch_size = left1.shape[0]

        out_put = net([left1,left2],[right1,right2],[head1,head2],[back1, back2],[map1,map2])  # [B, 7] , Quaternion
        print(out_put[0].shape)
        print(out_put[1].shape)
        time.sleep(2333)
