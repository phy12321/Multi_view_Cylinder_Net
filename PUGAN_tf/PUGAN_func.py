# -*- coding: gbk -*-
import tensorflow as tf
import logging
import pprint
import argparse
try:
    from PUGAN_tf.Upsampling.model import Model
except:
    from Upsampling.model import Model

import numpy as np
import random
pp = pprint.PrettyPrinter()
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from time import sleep

class PUGAN(object):
    def __init__(self, ckp_name='/home/phy12321/code/PU-GAN/log/20201212-1633'):
        parser_test = argparse.ArgumentParser("test")
        self.FLAGS = parser_test.parse_args()
        self.FLAGS.ckp_name = ckp_name
        self.FLAGS.more_up = 2
        self.FLAGS.up_ratio = 4
        self.FLAGS.num_point = 256
        self.FLAGS.patch_num_point = 256
        self.FLAGS.patch_num_ratio = 3
        self.FLAGS.use_non_uniform = True
        # self.FLAGS.log_dir = os.path.join(os.getcwd(), 'log', self.FLAGS.ckp_name)

        pp.pprint(self.FLAGS)

        self.run_config = tf.ConfigProto()
        self.run_config.gpu_options.allow_growth = True
        sess = tf.Session(config=self.run_config,)
        self.model = Model(self.FLAGS, sess)
        self.model.prepare_inference()

    def __call__(self, pc, seed_pt):
        """
        upsample points at the seed position
        :param pc: points
        :param seed_list: (n,3), n: num of patches, each seed will be sampled as a patch
        :return: unsampled points
        """
        print("number of seeds / patches: %d" % seed_pt.shape[0])
        input_list = []
        up_point_list = []
        patches = self.__extract_knn_patch(seed_pt, pc, self.FLAGS.patch_num_point)
        # seed = farthest_point_sample(seed1_num, points).eval()[0]

        for point in tqdm(patches, total=len(patches)):
            up_point = self.model.patch_prediction(point)
            up_point = np.squeeze(up_point, axis=0)
            input_list.append(point)
            up_point_list.append(up_point)
        return np.concatenate(input_list, axis=0), np.concatenate(up_point_list, axis=0)

    def __extract_knn_patch(self, queries, pc, k):
        knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
        knn_search.fit(pc)
        knn_idx = knn_search.kneighbors(queries, return_distance=False)
        k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
        return k_patches


def main(argv):
    up_sampler = PUGAN(ckp_name='/home/phy12321/code/PU-GAN/log/20201212-1633')
    pt = np.fromfile("/home/phy12321/dataset/kitti_odomentry/sequences/00/velodyne/000010.bin", count=-1,
                     dtype=np.float32).reshape([-1, 4])[:, :3]

    x = pt[:, 0]
    y = pt[:, 1]
    d = np.sqrt(x ** 2 + y ** 2)

    d_upsample_range = np.around(np.min(d) + 0.05 * (np.max(d) - np.min(d)), decimals=1)
    iop = np.argwhere(d < d_upsample_range)
    print(np.min(iop), np.max(iop))
    sleep(333)

    seed_list = np.array(random.sample(range(1, iop.shape[0]), 100))
    seed_pt = pt[seed_list, :3]
    input_list, up_point_list = projector(pt, seed_pt)
    display_points(input_list, up_point_list)
    sleep(20)


    # seed_list = random.sample(range(1,pt.shape[0]), 100)
    seed_pt = np.array([[2, 2, 1], [2, 2, 2], [3, 3, 2], [4, 4, 2]])

    input_list, up_point_list = up_sampler(pt, seed_pt)

    print(f"pt : {pt.shape}")
    print(f"input_list: {input_list.shape}")
    print(f"up_point_list: {up_point_list.shape}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
