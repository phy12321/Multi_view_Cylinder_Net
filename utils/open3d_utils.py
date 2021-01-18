import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch


def compute_normals(points):
    """

    :param points: np, (N, 3) or (N, 4)
    :return: (N,3),normals
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=10))
    normals = np.abs(np.asarray(pcd.normals))
    return normals


def display_points(source, template, transformed_source_from_template=None, title='points'):
    """
    :param source: numpy
    :param template: numpy
    :param transformed_source: numpy
    :return:
    """
    if source.shape[1] > 4:
        source = source.T
    if template.shape[1] > 4:
        template = template.T
    if transformed_source_from_template is not None:
        if transformed_source_from_template.shape[1] > 4:
            transformed_source_from_template = transformed_source_from_template.T
    source_ = o3d.geometry.PointCloud()
    source_.points = o3d.utility.Vector3dVector(source[:, :3])
    source_.paint_uniform_color([0.8, 0, 0])  # red

    template_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template[:, :3])
    template_.paint_uniform_color([0.6, 0.3, 1])  # purple

    if transformed_source_from_template is not None:
        transformed_source_from_template_ = o3d.geometry.PointCloud()
        transformed_source_from_template_.points = o3d.utility.Vector3dVector(transformed_source_from_template[:, :3])
        transformed_source_from_template_.paint_uniform_color([0, 1, 1])  # blue

        # axis_pcd = o3d.create_mesh_coordinate_frame(size=50, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([source_, template_, transformed_source_from_template_],
                                          window_name=title, width=3200,
                                          height=2400)  # , template_, transformed_source_from_template_])
    else:
        o3d.visualization.draw_geometries([source_, template_], window_name=title, width=800,
                                          height=600)  # red,purple,blue


def display_map(map1, map2=None, map3=None, map4=None, map5=None, axis=0):
    """

    :param map1:<class 'numpy.ndarray'> (72, 1016)
    :param map2:<class 'numpy.ndarray'> (72, 1016)
    :return:
    """
    map_list = [map1, map2, map3, map4, map5]
    n = 0
    for map in map_list:
        if map is not None:
            n += 1
    if axis == 1:
        plt.figure(figsize=(12 * n, 5))
    elif axis == 0:
        plt.figure(figsize=(48, 4 * n))  # w,h
    else:
        raise ValueError("wrong axis")

    for i in range(n):
        if axis == 1:
            plt.subplot(1, n, i + 1)
        elif axis == 0:
            plt.subplot(n, 1, i + 1)

        if torch.is_tensor(map_list[i]):
            plt.imshow(map_list[i].detach().cpu().numpy())
            print(f"shape: {map_list[i].shape}, max-min: {torch.max(map_list[i]), torch.min(map_list[i])} ")
        else:
            plt.imshow(np.array(map_list[i]))
            print(f"shape: {map_list[i].shape}, max-min: {np.max(map_list[i]), np.min(map_list[i])} ")
        plt.axis('off')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def display_bev(map_list):
    num_img = len(map_list)
    plt.figure()
    for i in range(num_img):
        plt.subplot(1, num_img, i + 1)
        plt.imshow(map_list[i])
        print(np.min(map_list[i]), np.max(map_list[i]))
        plt.axis('off')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
