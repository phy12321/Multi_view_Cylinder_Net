# -*-coding:utf-8-*-
import torch
import numpy as np
import os
from utils import kitti_data_set
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from utils.cupy_utils import CupyPhotoMetricCriterion
from model.net import Net
from utils.data_utils import get_logger
from utils.tensor_utils import pose_L2_loss
import torch.optim as optim
from tensorboardX import SummaryWriter
import shutil
from utils.msssim_utils import msssim
from args import arg_list
import cupy as cp

debug = arg_list.debug
on_aigo = arg_list.on_aigo

if on_aigo:
    arg_list.data_set = os.path.join("dataset", "kitti")
    arg_list.log_path = "/home/phy12321/code/Multi_view_Cylinder_Net"
else:
    arg_list.data_set = "/data/phy12321/kitti_lidar_odomentry"
    arg_list.log_path = "/out"

start_time = time.strftime('%Y_%m_%d-%H-%M', time.localtime(time.time()))
log_base_path = os.path.join(arg_list.log_path, "Logs", str(start_time))
code_short_cut_dir = os.path.join(log_base_path, "code_short_cut")
ckp_path = os.path.join(log_base_path, "checkpoints")
tf_log_path = os.path.join(log_base_path, "tensorboard")
for path in [log_base_path, ckp_path, tf_log_path, code_short_cut_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

logger = get_logger(log_base_path)
tf_writer = SummaryWriter(tf_log_path)


def log_string(s):
    logger.info(s)
    print(s)


log_string(f"\n###############################################  Note  ##############################################\n"
           f"remove the detach() operation of reprojection loss.\n"
           f"init lr = 0.001, milestones = [100,180], epoch = 300\n"
           f"remove msssim loss, alpha = 0, pose_loss_norm = 2 , reproj_range_loss_w= 0.5\n" 
           f"single net + deep rot FC + deep trans FC\n"
           f"############################################# End Of Note ############################################\n")


def bkp_file():
    # copy code_file bkp
    shutil.copy(os.path.abspath(__file__), os.path.join(code_short_cut_dir, "train_bkp.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "args.py"), os.path.join(code_short_cut_dir, "args_bkp.py"))

    shutil.copy(os.path.join(os.path.dirname(__file__), "model", "net.py"),
                os.path.join(code_short_cut_dir, "net_bkp.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "model", "model_zoo.py"),
                os.path.join(code_short_cut_dir, "model_zoo_bkp.py"))

    shutil.copy(os.path.join(os.path.dirname(__file__), "utils", "kitti_data_set.py"),
                os.path.join(code_short_cut_dir, "kitti_data_set.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "utils", "data_utils.py"),
                os.path.join(code_short_cut_dir, "data_utils_bkp.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "utils", "pc_utils.py"),
                os.path.join(code_short_cut_dir, "pc_utils_bkp.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "utils", "criterion.py"),
                os.path.join(code_short_cut_dir, "criterion_bkp.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "utils", "cupy_utils.py"),
                os.path.join(code_short_cut_dir, "cupy_utils_bkp.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "utils", "tensor_utils.py"),
                os.path.join(code_short_cut_dir, "tensor_utils_bkp.py"))


def load_ckp(net):
    # load checkpoint:
    if arg_list.ckp is not None:
        net_stat = torch.load(arg_list.ckp, map_location='cpu')
        net.load_state_dict(net_stat['net'])
        ckp_epoch = net_stat['epoch']
        log_string(f"load ckp from epoch {ckp_epoch}, {arg_list.ckp}")
    else:
        ckp_epoch = 0
        log_string("Train from scratch. ")

    if torch.cuda.is_available():
        net = net.cuda()

    return net, ckp_epoch


def get_model_parameters(net):
    weight_decay_list = (param for name, param in net.named_parameters() if
                         name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    log_string(f'{total_params / 1000000}M  total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log_string(f'{total_trainable_params / 1000000}M  training parameters.')

    return parameters


def get_dataloader():
    data_path = arg_list.data_set
    train_dataset = kitti_data_set.KittiDataset(data_path, mode="train", seq_list=arg_list.train_seq,
                                                train_num_per_class=arg_list.train_num_per_class,
                                                rot_form=arg_list.rot_form)

    test_dataset = kitti_data_set.KittiDataset(data_path, mode="test", seq_list=arg_list.test_seq,
                                               test_num=arg_list.test_num,
                                               rot_form=arg_list.rot_form)

    train_dataloader = DataLoader(train_dataset, batch_size=arg_list.batch_size,
                                  shuffle=arg_list.shuffle, num_workers=arg_list.num_workers, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=arg_list.batch_size,
                                 shuffle=arg_list.shuffle, num_workers=arg_list.num_workers, pin_memory=True)

    return train_dataloader, test_dataloader


def train_one_epoch(train_dataloader, net, opt, iter_num):
    num_examples = 0
    epoch_loss = 0
    epoch_pose_loss = 0
    epoch_pixel_loss = 0
    step = 0
    opt.zero_grad()
    photo_metric_criterion = CupyPhotoMetricCriterion()
    # cp.cuda.Device(0).use()
    alpha = torch.tensor(arg_list.img_alpha).cuda()

    for data in train_dataloader:

        img1, img2, rel_T_12, src_path, tgt_path = data  # [l,r,h,b][l,r,h,b][rot,trans]
        left1, right1, head1, back1, map1 = img1[0].cuda(), img1[1].cuda(), \
                                            img1[2].cuda(), img1[3].cuda(), img1[4].cuda()
        left2, right2, head2, back2, map2 = img2[0].cuda(), img2[1].cuda(), \
                                            img2[2].cuda(), img2[3].cuda(), img2[4].cuda()

        rot_gt, trans_gt = rel_T_12[0].cuda(), rel_T_12[1].cuda()
        batch_size = left1.shape[0]

        net.train()
        rot_pred, trans_pred = net([left1, left2], [right1, right2],
                                   [head1, head2], [back1, back2],
                                   [map1, map2])  # [B, 7] ,

        ###########################################################################################################
        # pose_loss:  quaternion , euler, flatten matrix
        rot_loss, trans_loss = pose_L2_loss([rot_pred, trans_pred],
                                            [rot_gt, trans_gt], arg_list.pose_loss_norm)

        pose_loss = arg_list.rot_loss_w * rot_loss + arg_list.trans_loss_w * trans_loss

        ##########################################################################################################
        # reprojection_loss:
        error_map = torch.zeros(batch_size, 1, map1.shape[-2], map1.shape[-1]).cuda()
        error_map_gt = torch.zeros(batch_size, 1, map1.shape[-2], map1.shape[-1]).cuda()
        for i in range(batch_size):
            tgt_depth = map1[i, 0, :, :]
            masked_error, masked_error_gt = photo_metric_criterion.apply(src_path[i],
                                                                         [rot_pred[i], trans_pred[i]],
                                                                         [rot_gt[i], trans_gt[i]],
                                                                         tgt_depth)
            error_map[i, 0, :, :] = masked_error
            error_map_gt[i, 0, :, :] = masked_error_gt
        # use mean_sub loss instead of sub_mean loss

        pixel_loss = (torch.abs(torch.mean(error_map) - torch.mean(error_map_gt)))
        # print(f"error: {torch.mean(error_map)}; error_gt: {torch.mean(error_map_gt)};"
        #       f" diff: {pixel_loss}, pose_loss : {pose_loss}")
        # msssim_loss = msssim(error_map, error_map_gt, normalize="Relu")
        img_loss = (1 - alpha) * pixel_loss  # + (alpha * 0.5 * (1 - msssim_loss))

        loss = arg_list.pose_loss_w * pose_loss + arg_list.reproj_range_loss_w * img_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        num_examples += batch_size
        epoch_loss += loss.item() * batch_size
        epoch_pose_loss += pose_loss.item() * batch_size
        epoch_pixel_loss += pixel_loss.item() * batch_size

        # if loss.item() > 10000:
        #     log_string('Unconvergence !' + str(loss.item()))
        #     time.sleep(3000)
        step += 1
        if step % arg_list.display_step == 0:
            print(f'step{step}/{len(train_dataloader)};\ttotal / pose / pixel_msssim : '
                  + f'{np.around(loss.item(), 2)} / '
                  + f'{np.around(pose_loss.item() * arg_list.pose_loss_w, 2)} / '
                  + f"{np.around(img_loss.item() * arg_list.reproj_range_loss_w, 2)} ")
            tf_writer.add_scalar('train_loss', loss.item(), global_step=iter_num + step)
            tf_writer.add_scalar('train_pose_loss', pose_loss.item(),
                                 global_step=iter_num + step)
            tf_writer.add_scalar('train_reproj_range_loss', img_loss.item(),
                                 global_step=iter_num + step)

    return epoch_loss * 1.0 / num_examples, epoch_pose_loss * 1.0 / num_examples, \
           epoch_pixel_loss * 1.0 / num_examples


def test_one_epoch(test_dataloader, net):
    num_examples = 0
    epoch_loss = 0
    epoch_pose_loss = 0
    epoch_pixel_loss = 0
    photo_metric_criterion = CupyPhotoMetricCriterion()
    alpha = torch.tensor(arg_list.img_alpha).cuda()

    for data in tqdm(test_dataloader,
                     total=len(test_dataloader)):  # tqdm(train_dataloader, total=len(train_dataloader)):

        img1, img2, rel_T_12, src_path, tgt_path = data  # [l,r,h,b][l,r,h,b][rot,trans]
        left1, right1, head1, back1, map1 = img1[0].cuda(), img1[1].cuda(), \
                                            img1[2].cuda(), img1[3].cuda(), img1[4].cuda()
        left2, right2, head2, back2, map2 = img2[0].cuda(), img2[1].cuda(), \
                                            img2[2].cuda(), img2[3].cuda(), img2[4].cuda()

        rot_gt, trans_gt = rel_T_12[0].cuda(), rel_T_12[1].cuda()
        batch_size = left1.shape[0]

        net.train()
        rot_pred, trans_pred = net([left1, left2], [right1, right2],
                                   [head1, head2], [back1, back2],
                                   [map1, map2])  # [B, 7] ,

        ###########################################################################################################
        # pose_loss:  quaternion , euler, flatten matrix
        rot_loss, trans_loss = pose_L2_loss([rot_pred, trans_pred],
                                            [rot_gt, trans_gt], arg_list.pose_loss_norm)

        pose_loss = arg_list.rot_loss_w * rot_loss + arg_list.trans_loss_w * trans_loss

        ##########################################################################################################
        # reprojection_loss:
        error_map = torch.zeros(batch_size, 1, map1.shape[-2], map1.shape[-1]).cuda()
        error_map_gt = torch.zeros(batch_size, 1, map1.shape[-2], map1.shape[-1]).cuda()
        for i in range(batch_size):
            tgt_depth = map1[i, 0, :, :]
            masked_error, masked_error_gt = photo_metric_criterion.apply(src_path[i],
                                                                         [rot_pred[i], trans_pred[i]],
                                                                         [rot_gt[i], trans_gt[i]],
                                                                         tgt_depth)
            error_map[i, 0, :, :] = masked_error
            error_map_gt[i, 0, :, :] = masked_error_gt

        # use mean_sub loss instead of sub_mean loss
        pixel_loss = (torch.abs(torch.mean(error_map) - torch.mean(error_map_gt)))
        # print(f"error: {torch.mean(error_map)* 10}; error_gt: {torch.mean(error_map_gt)* 10};"
        #       f" diff: {pixel_loss * 10}")
        msssim_loss = msssim(error_map, error_map_gt, normalize="Relu")
        img_loss = (1 - alpha) * pixel_loss + (alpha * 0.5 * (1 - msssim_loss))

        loss = arg_list.pose_loss_w * pose_loss + arg_list.reproj_range_loss_w * img_loss

        num_examples += batch_size
        epoch_loss += loss.item() * batch_size
        epoch_pose_loss += pose_loss.item() * batch_size
        epoch_pixel_loss += pixel_loss.item() * batch_size

    return epoch_loss * 1.0 / num_examples, \
           epoch_pose_loss * 1.0 / num_examples, \
           epoch_pixel_loss * 1.0 / num_examples


def main():
    if debug:
        log_string("Debug mode:")
        arg_list.train_num_per_class = (40, 40, 40)
        arg_list.test_num = 100
        arg_list.epoch = 2
        arg_list.train_seq = ['04']
        arg_list.test_seq = ['04']
        arg_list.batch_size = 8
        arg_list.num_workers = 2
    # make file shortcut
    log_string(f"\n##########################################  Args List  ###########################################\n"
               + f"{arg_list}")

    bkp_file()

    # dataloader
    train_dataloader, test_dataloader = get_dataloader()

    # model
    if arg_list.rot_form == "euler":
        rot_channel = 3
    elif arg_list.rot_form == "matrix":
        rot_channel = 9
    elif arg_list.rot_form == "quaternion":
        rot_channel = 4
    else:
        raise ValueError(f"wrong rot form: {arg_list.rot_form}")
    net = Net(in_channel=2, rot_channel=rot_channel)
    net, ckp_epoch = load_ckp(net)
    parameters = get_model_parameters(net)

    # optimizer
    if arg_list.optimizer == "SGD":
        log_string("use SGD as optimizer")
        opt = optim.SGD(parameters, lr=arg_list.lr, momentum=arg_list.momentum, weight_decay=arg_list.weight_decay)
    if arg_list.optimizer == "Adam":
        log_string("use Adam as optimizer")
        opt = optim.Adam(parameters, lr=arg_list.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        raise ValueError(" illegal optimizer ! ")

    if arg_list.lr_scheduler == "steplr":
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        log_string("use step lr scheduler")
    elif arg_list.lr_scheduler == "autolr":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1,
                                                            patience=3, verbose=True, threshold=0.001)
        log_string("use auto lr scheduler")
    elif arg_list.lr_scheduler == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt,
                                                      milestones=arg_list.milestones,
                                                      gamma=arg_list.divid_factor)
        log_string("use Multi-Step lr scheduler")

    log_string("use fixed weighted sum loss")

    #####################################################
    # start training :
    ####################################################

    log_string(("train_seq:", arg_list.train_seq,
                "; number of train_data:" + str(train_dataloader.__len__() * arg_list.batch_size)))
    log_string(("test_seq:", arg_list.test_seq,
                " number of test_data:" + str(test_dataloader.__len__() * arg_list.batch_size)))
    log_string("\n##########################################  END of Args List ######################################\n"
               + "\n")
    best_test_loss = np.inf

    total_epoch = arg_list.epoch - ckp_epoch
    best_epoch = 0
    start = time.time()
    iter_num = 0
    for e in range(1, total_epoch + 1):
        s = time.time()
        epoch = ckp_epoch + e
        log_string("**** Epoch:" + str(epoch) + '/' + str(arg_list.epoch))

        if e == 1:
            arg_list.display_step = 5
        else:
            arg_list.display_step = 100
        train_loss, pose_loss, pixel_loss = train_one_epoch(train_dataloader, net, opt, iter_num)
        iter_num += len(train_dataloader)
        log_string(f" *  train_loss / train_pose_loss / train_pixel_loss:"
                   + str(np.around(train_loss, 4)) + ' / '
                   + str(np.around(pose_loss, 4)) + ' / '
                   + str(np.around(pixel_loss, 4)))

        # eval :
        with torch.no_grad():
            test_loss, test_pose_loss, test_pixel_loss = test_one_epoch(test_dataloader, net.eval())

            log_string(" *  test_loss / test_pose_loss / test_pixel_loss: "
                       + str(np.around(test_loss, 4)) + ' / '
                       + str(np.around(test_pose_loss, 4)) + ' / '
                       + str(np.around(test_pixel_loss, 4)))

            tf_writer.add_scalar('test_loss', test_loss, global_step=epoch)
            tf_writer.add_scalar('test_pose_loss', test_pose_loss, global_step=epoch)
            tf_writer.add_scalar('test_pixel_loss', test_pixel_loss, global_step=epoch)

            state = {'net': net.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch}
            if best_test_loss > test_loss:
                best_test_loss = test_loss
                torch.save(state, os.path.join(ckp_path, 'best.pth'))
                best_epoch = epoch

            log_string(f"current best epoch : {best_epoch}")
            # if 0 == (epoch) % 10:
            #     torch.save(state, os.path.join(ckp_path, 'model_epoch_%d.pth' % epoch ))

        if arg_list.lr_scheduler == "steplr":
            lr_scheduler.step()
            log_string(f"learning_rate = {lr_scheduler.get_lr()}")

        elif arg_list.lr_scheduler == "autolr":
            lr_scheduler.step(train_loss)
        elif arg_list.lr_scheduler == "MultiStepLR":
            lr_scheduler.step()
            log_string(f"learning_rate = {lr_scheduler.get_last_lr()}")

        log_string("using time: " + str(np.around((time.time() - s) / 60, 4)) + " min")

    log_string("Training finished . ")
    consume = time.time() - start
    min_left = np.floor(consume / 60) - (np.floor(consume / 3600) * 60)
    time_per_epoch = np.floor(consume / (total_epoch * 60))
    log_string("Using time: " + str(int(np.floor(consume / 3600))) + " h " + str(int(min_left)) + " min")
    log_string("each epoch using time: " + str(int(time_per_epoch)) + " min")


if __name__ == "__main__":

    torch.cuda.empty_cache()
    main()
    # test:
    torch.cuda.empty_cache()
    log_string(f"Start Test.")
    if on_aigo:
        os.system(f"python test.py --log_date '{start_time}'")
    else:
        os.system(f"python /data/phy12321/code/test.py --log_date '{start_time}'")
    log_string(f"Test finished.")

## tensorboard --logdir=/home/phy12321/code/lidar_points_process/Logs/2020_10_10-22-00+/tensorboard
