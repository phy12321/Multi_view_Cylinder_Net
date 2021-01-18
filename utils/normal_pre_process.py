import numpy as np
import os
import time
import open3d as o3d
from tqdm import tqdm

def compute_normals(points, radius=10, max_nn=10, dtype=np.float32):
    """
    :param points: np, (N, 3) or (N, 4)
    :return: (N,3),normals
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.abs(np.asarray(pcd.normals))
    return normals.astype(dtype)

seq_list = ['04']

data_path = os.path.join(os.getcwd(),"..","dataset","kitti","sequences")

for seq in seq_list:
    print(f"processing {seq}",end=",")
    normal_file = os.path.join(data_path, seq, "normal")
    if not os.path.exists(normal_file):
        os.makedirs(normal_file)
    point_list = os.listdir(os.path.join(data_path, seq,'velodyne'))
    point_list.sort(key=lambda x:int(x[:-4]))
    for point_name in tqdm(point_list):
        path = os.path.join(data_path, seq, 'velodyne',point_name)
        save_path = os.path.join(normal_file,  point_name)
        pt = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
        normal = compute_normals(pt)
        normal.tofile(save_path)
print(f"normals are saved to path； {data_path}")




























