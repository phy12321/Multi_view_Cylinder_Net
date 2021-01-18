import argparse

args = argparse.ArgumentParser("multi-view-Cylinder-Net")
args.add_argument('--batch_size', type=int, default=48)
args.add_argument('--data_set', type=str, default='dataset/kitti')  # dataset/kitti
args.add_argument('--log_path', type=str, default='/out')
args.add_argument('--display_step', type=int)
args.add_argument('--shuffle', type=bool, default=True)
args.add_argument('--num_workers', type=int, default=24)
args.add_argument('--epoch', type=int, default=300)
args.add_argument('--optimizer', type=str, default='Adam',
                  choices=["Adam", "SGD"])
args.add_argument('--lr_scheduler', type=str, default="MultiStepLR",
                  choices=["steplr", "autolr", "MultiStepLR"])
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--milestones', type=list, default=[100, 180])
args.add_argument('--divid_factor', type=float, default=0.5)
args.add_argument('--momentum', type=float, default=0.9)
args.add_argument('--rot_form', type=str, default="euler", choices=["matrix", "quaternion", "euler"])
#  loss weights
args.add_argument('--img_alpha', type=float, default=0)
args.add_argument('--trans_loss_w', type=float, default=1)
args.add_argument('--rot_loss_w', type=float, default=10)
args.add_argument('--pose_loss_w', type=float, default=1)
args.add_argument('--reproj_range_loss_w', type=float, default=.5)
args.add_argument('--reproj_normal_loss_w', type=float, default=.1)
args.add_argument('--pose_loss_norm', type=int, default=2)
args.add_argument('--weight_decay', type=float, default=5e-3)
# dataset
args.add_argument('--train_seq', type=list, default=None)  # all: ['00', '01', '02', '03', '04', '05', '06', '07']
args.add_argument('--train_num_per_class', type=tuple, default=(8000, 8000, 8000))  # 6000
args.add_argument('--test_seq', type=list, default=None)  # all: ["08","09","10"]
args.add_argument('--test_num', type=int, default=2000)
args.add_argument('--overlap', type=int, default=20)  # overlap of multi view map
args.add_argument('--ckp', type=str, default=None)
# default="/home/phy12321/code/Multi_view_Cylinder_Net/Logs/2020_12_25-15-33/checkpoints/best.pth")
##  debug
args.add_argument('--debug', type=bool, default=False)
args.add_argument('--on_aigo', type=bool, default=True)

arg_list = args.parse_args()
