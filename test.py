import torch
import time
import os
from utils import kitti_data_set
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_utils import matrix_rot_trans_error, Rel_Pose2Abs_Pose, slp
import numpy as np
import shutil
from evo.evaluation import kittiOdomEval
import subprocess


test_args = argparse.ArgumentParser("kitti_evo")
test_args.add_argument('--log_date', type=str, default="2020_12_18-12-42")
test_args.add_argument('--seq_list', type=list, default=['00', '01', '02', '03', '04', '05', '06', '07','08','09','10'])
test_args.add_argument('--data_set', type=str, default='dataset/kitti')
test_args.add_argument('--test_seq', type=bool, default=True)
test_args.add_argument('--evo_only', type=bool, default=False)

arg_list_test = test_args.parse_args()

log_path = os.path.join(os.getcwd(), 'Logs', arg_list_test.log_date)
model_path = os.path.join(log_path, 'code_short_cut', "net_bkp.py")
result_path = os.path.join(log_path, 'results')

def load_net(ckp_path):
    shutil.copy(model_path, os.getcwd())
    from net_bkp import Net
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    net = Net(in_channel=2, rot_channel=3)
    net.load_state_dict(ckp['net'])
    net.eval()
    print("load ckp from " + ckp_path + ", epoch ", ckp['epoch'])
    os.remove(os.path.join(os.getcwd(), "net_bkp.py"))
    return net

def nn(number, aqc=6):
    return np.round(np.array(number), aqc)

def eval_error(net, test_dataloader, eval_dataset=False):
    """

    :param net:
    :param test_dataloader:
    :return:
    """
    num_examples = 0
    m_angular_error1 = 0, 0, 0
    m_trans_error1 = 0, 0, 0
    pred_pose = []
    with torch.no_grad():
        start = time.time()
        for data in tqdm(test_dataloader, total=len(test_dataloader)):

            img1, img2, rel_T_12, src_path, tgt_path = data  ## [l,r,h,b][l,r,h,b][rot,trans]
            left1, right1, head1, back1, map1 = img1[0].cuda(), img1[1].cuda(), \
                                                img1[2].cuda(), img1[3].cuda(), img1[4].cuda()
            left2, right2, head2, back2, map2 = img2[0].cuda(), img2[1].cuda(), \
                                                img2[2].cuda(), img2[3].cuda(), img2[4].cuda()

            # rot_gt, trans_gt = rel_T_12[0].cuda(), rel_T_12[1].cuda()
            batch_size = left1.shape[0]
            assert left1.shape[0] == 1
            num_examples += batch_size

            rot_pred, trans_pred = net([left1, left2], [right1, right2],
                                       [head1, head2], [back1, back2],
                                       [map1, map2])  # [B, 7]
            rot_pred = rot_pred.cpu()
            trans_pred = trans_pred.cpu()
            pred = np.concatenate([rot_pred, trans_pred], axis=1)
            pred_pose.append(np.squeeze(pred))

            # abs_angular_error and metre_error
            angular_error1, trans_error1 = matrix_rot_trans_error([rot_pred, trans_pred], rel_T_12)

            m_angular_error1 += angular_error1.sum()
            m_trans_error1 += trans_error1.sum()

        m_angular_error1 = m_angular_error1 * 1.0 / num_examples
        m_trans_error1 = m_trans_error1 * 1.0 / num_examples
        consume_per_sample = (time.time() - start) / num_examples

    aqc = 5
    m_angular_error1 = np.around(m_angular_error1, aqc)
    m_trans_error1 = np.around(m_trans_error1, aqc)
    consume_per_sample = np.around(consume_per_sample, aqc)

    return m_angular_error1, m_trans_error1, consume_per_sample, pred_pose

def evo_result():
    os.system(f"dos2unix evo/evo.sh")
    os.system(f"sed -i 's/\r//' evo/evo.sh")
    os.putenv("PATH", "/home/phy12321/anaconda3/envs/torch17/bin:/home/phy12321/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin")
    for seq in arg_list_test.seq_list:
        subprocess.call(f"bash ./evo/evo.sh {arg_list_test.log_date} {seq}", shell=True)

def kitti_evo_tools():
    gt_pose_path = os.path.join(arg_list_test.data_set,'poses')
    pose_eval = kittiOdomEval(gt_pose_path, result_path, eval_seqs=arg_list_test.seq_list)
    pose_eval.eval(toCameraCoord=False)

if __name__ == "__main__":
    print(f"arg_list_test.test_seq = {arg_list_test.test_seq}, arg_list_test.evo_only = {arg_list_test.evo_only}")
    assert arg_list_test.test_seq + arg_list_test.evo_only == 1

    if arg_list_test.evo_only:
        print(f"only evo the pred pose of {arg_list_test.log_date}")
        kitti_evo_tools()
        exit(0)

    if not os.path.exists(result_path):
        os.makedirs(result_path)


    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), arg_list_test.data_set)
    ckp_path = os.path.join(log_path, 'checkpoints', 'best.pth')
    net = load_net(ckp_path)
    if torch.cuda.is_available():
        net = net.cuda()

    for seq in arg_list_test.seq_list:
        test_result_path = os.path.join(result_path, f'angular_error_{str(seq)}.log')
        pred_pose_path = os.path.join(result_path, f'pred_pose_{str(seq)}.txt')
        calib_path = os.path.join(os.getcwd(), 'dataset/kitti', 'sequences', seq, 'calib.txt')
        Tr = np.array(np.loadtxt(calib_path, dtype='str')[-1][1:], dtype=np.float64)
        test_dataset = kitti_data_set.KittiDataset(data_path=data_path, rot_form="matrix",
                                                   mode="test",seq_list=[seq])

        test_dataloader = DataLoader(test_dataset, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     pin_memory=True, drop_last=True)

        m_angular_error, m_trans_error, consume_per_sample, pred_pose = eval_error(net.eval(), test_dataloader)
        with open(test_result_path, "w+") as f:
            f.write(f"DeepICP: \t0.056° / 0.018m\n\n")
            f.write(f"1:\t{str(m_angular_error[0])}° / {str(m_trans_error[0])}m; ")
            f.write(f" 相比DeepICP增大:{str(nn(m_angular_error[0] - 0.056, 4))}°;")
            f.write(f"{str(nn(m_trans_error[0] - 0.018, 4))}m\n")

            f.write("\ntime_per_sample: " + str(consume_per_sample) + ' s')
        print(f"    Eval_results is saved to {test_result_path}")

        abs_pose = Rel_Pose2Abs_Pose(pred_pose, calib=Tr)
        np.savetxt(pred_pose_path, abs_pose, fmt='%f', delimiter=' ')
        print(f"    Pose_list is saved to {pred_pose_path}\n")

    kitti_evo_tools()
