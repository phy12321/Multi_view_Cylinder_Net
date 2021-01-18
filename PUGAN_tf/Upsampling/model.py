# -*- coding: gbk -*-
# @Time        : 16/1/2019 5:04 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
import tensorflow as tf
try:
    from PUGAN_tf.Common import model_utils
    from PUGAN_tf.Upsampling.generator import Generator
    from PUGAN_tf.Common import pc_util
    from PUGAN_tf.tf_ops.sampling.tf_sampling import farthest_point_sample
except:
    from Common import model_utils
    from Upsampling.generator import Generator
    from Common import pc_util
    from tf_ops.sampling.tf_sampling import farthest_point_sample
import os
from tqdm import tqdm
import math
from time import time, sleep
from termcolor import colored
import open3d as o3d
import numpy as np


class Model(object):
    def __init__(self, opts, sess):
        self.sess = sess
        self.opts = opts

    def prepare_inference(self):
        self.inputs = tf.placeholder(tf.float32, shape=[1, self.opts.patch_num_point, 3])
        is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        Gen = Generator(self.opts, is_training, name='generator')
        self.pred_pc = Gen(self.inputs)
        for i in range(round(math.pow(self.opts.up_ratio, 1 / 4)) - 1):
            self.pred_pc = Gen(self.pred_pc)
        saver = tf.train.Saver()
        restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.ckp_name)
        saver.restore(self.sess, checkpoint_path)

    def patch_prediction(self, patch_point):
        # normalize the point clouds
        patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
        patch_point = np.expand_dims(patch_point, axis=0)
        pred = self.sess.run([self.pred_pc], feed_dict={self.inputs: patch_point})
        pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
        return pred

    def pc_prediction(self, pc):
        ## get  seed
        points = tf.convert_to_tensor(np.expand_dims(pc, axis=0), dtype=tf.float32)
        start = time()
        print(f'num_point: {pc.shape}')
        print('patch_num_point:', self.opts.patch_num_point)
        seed1_num = 200 #int(pc.shape[0] / self.opts.patch_num_point * self.opts.patch_num_ratio)

        ## FPS sampling
        seed = farthest_point_sample(seed1_num, points).eval()[0]
        seed_list = seed[:seed1_num]
        print(f"seed_length: {seed.shape}, using {seed_list.shape}")
        print("farthest distance sampling cost", time() - start)
        print("number of patches: %d" % len(seed_list))
        input_list = []
        up_point_list = []
        ## get patch
        patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, self.opts.patch_num_point)

        print(f"patches.shape : {patches.shape}")
        for point in tqdm(patches, total=len(patches)):
            print(f"patch: {point.shape}")  # 256
            up_point = self.patch_prediction(point)
            up_point = np.squeeze(up_point, axis=0)
            print(f"up_point: {up_point.shape}")  # 1024
            input_list.append(point)
            up_point_list.append(up_point)
        return input_list, up_point_list

    def dis_points(self, pred_pc):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pred_pc)
        point_cloud.paint_uniform_color([np.random.random(1), np.random.random(1), np.random.random(1)])  # purple
        o3d.visualization.draw_geometries([point_cloud])
        print(f"display_pc.shape:{pred_pc.shape}")

    def inference_one_pt(self, point_path):
        out_point_num = int(self.opts.num_point * self.opts.up_ratio)
        pc = pc_util.load(point_path)[:, :3]
        # self.dis_points(pc)
        pc, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)

        start = time()
        input_list, pred_list = self.pc_prediction(pc)
        end = time()
        print("one sample time: ", end - start)
        pred_pc = np.concatenate(pred_list, axis=0)
        pred_pc = (pred_pc * furthest_distance) + centroid
        pred_pc = np.reshape(pred_pc, [-1, 3])
        # self.dis_points(pred_pc)
        pred_pc = pred_pc[:, :]
        idx = farthest_point_sample(out_point_num, pred_pc[np.newaxis, ...]).eval()[0]
        pred_pc = pred_pc[idx, 0:3]
        path = os.path.join(self.opts.out_folder, point_path.split('/')[-1][:-4] + '.ply')
        print(f"save_path, {path}")
        np.savetxt(path[:-4] + '.bin', pred_pc, fmt='%.6f')
