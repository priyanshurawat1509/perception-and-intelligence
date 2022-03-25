#!/usr/bin/env python
# coding: utf-8

# In[2]:


##### Visualizing Camera image using opencv #####

import numpy as np
import cv2
import sys

img = cv2.imread("/users/priya/data/sets/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg", 0)
window_name = 'image'
cv2.imshow(window_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[35]:


##### Visualizing and colorizing lidar points by height and intensity  #####

import open3d as o3d
import numpy as np

seg_name='/users/priya/lidarseg/v1.0-mini/4484110755904050a880043268149497_lidarseg.bin'
seg=np.fromfile(seg_name, dtype=np.int8)

pcd_name='/users/priya/data/sets/nuscenes/samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489296047917.pcd.bin'
scan=np.fromfile(pcd_name, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

o3d.visualization.draw_geometries([pcd])


# In[36]:


##### Visualizing liDAR point cloud data and Colorizing LiDAR points by semantic label #####

import open3d as o3d
import numpy as np

seg_name='/users/priya/lidarseg/v1.0-mini/4484110755904050a880043268149497_lidarseg.bin'
seg=np.fromfile(seg_name, dtype=np.int8)

color = np.zeros([len(seg), 3])
color[:, 0] = seg/32
color[:, 1] = seg/32
color[:, 2] = seg/32

pcd_name='/users/priya/data/sets/nuscenes/samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489296047917.pcd.bin'
scan=np.fromfile(pcd_name, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([pcd])


# In[7]:


##### Visualizing Radar data points #####

# c. Using Open3D Library

pcd = o3d.io.read_point_cloud("/users/priya/data/sets/nuscenes/samples/RADAR_FRONT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT__1533151603555991.pcd")
o3d.visualization.draw_geometries([pcd])


# In[27]:


##### Visualizing Radar data points #####

# d. Colorizing on the basis of velocity

import struct
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

meta = []
with open("/users/priya/data/sets/nuscenes/samples/RADAR_FRONT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT__1533151603555991.pcd", 'rb') as f:
    for line in f:
        line = line.strip().decode('utf-8')
        meta.append(line)
        if line.startswith('DATA'):
            break

    data_binary = f.read()

# Get the header rows and check if they appear as expected.
assert meta[0].startswith('#'), 'First line must be comment'
assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
sizes = meta[3].split(' ')[1:]
types = meta[4].split(' ')[1:]
counts = meta[5].split(' ')[1:]
width = int(meta[6].split(' ')[1])
height = int(meta[7].split(' ')[1])
data = meta[10].split(' ')[1]
feature_count = len(types)
assert width > 0
assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
assert height == 1, 'Error: height != 0 not supported!'
assert data == 'binary'

# Lookup table for how to decode the binaries.
unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                     'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                     'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

# Decode each point.
offset = 0
point_count = width
points = []
for i in range(point_count):
    point = []
    for p in range(feature_count):
        start_p = offset
        end_p = start_p + int(sizes[p])
        assert end_p < len(data_binary)
        point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
        point.append(point_p)
        offset = end_p
    points.append(point)

# A NaN in the first point indicates an empty pointcloud.
point = np.array(points[0])

#return cls(np.zeros((feature_count, 0)))
print(np.shape(points))
points = np.asarray(points)

#coloring 
color = np.zeros([len(points[:,8]), 3])
color[:, 0] = points[:,8]/((np.max(points[:,8]))-(np.min(points[:,8])))*1
color[:, 1] = points[:,8]/((np.max(points[:,8]))-(np.min(points[:,8])))*2
color[:, 2] = points[:,8]/((np.max(points[:,8]))-(np.min(points[:,8])))*3

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])


# In[23]:


##### Question5.1. Visualizing Radar data projection on image and Calibration information #####

get_ipython().run_line_magic('matplotlib', 'inline')
from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import NuScenesExplorer

import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats,     get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap

nusc = NuScenes(version='v1.0-mini', dataroot='/users/priya/data/sets/nuscenes/', verbose=True)

filter_lidarseg_labels = []
lidarseg_preds_bin_path= ''

def map_pointcloud_to_image(pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0,
                                render_intensity: bool = False,
                                show_lidarseg: bool = False,
                                filter_lidarseg_labels: list = None,
                                lidarseg_preds_bin_path: str = None):
    
        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            if show_lidarseg:
                assert hasattr(NuScenes.nusc, 'lidarseg'), 'Error: nuScenes-lidarseg not installed!'

                # Ensure that lidar pointcloud is from a keyframe.
                assert pointsensor['is_key_frame'],                     'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                assert not render_intensity, 'Error: Invalid options selected. You can only select either '                                              'render_intensity or show_lidarseg, not both.'

            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(nusc.dataroot, cam['filename']))

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        print(cs_record)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        if render_intensity:
            assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, '                                                               'not %s!' % pointsensor['sensor_modality']
            # Retrieve the color from the intensities.
            # Performs arbitary scaling to achieve more visually pleasing results.
            intensities = pc.points[3, :]
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities = intensities ** 0.1
            intensities = np.maximum(0, intensities - 0.5)
            coloring = intensities
        elif show_lidarseg:
            assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, '                                                               'not %s!' % pointsensor['sensor_modality']

            if lidarseg_preds_bin_path:
                sample_token = nusc.get('sample_data', pointsensor_token)['sample_token']
                lidarseg_labels_filename = lidarseg_preds_bin_path
                assert os.path.exists(lidarseg_labels_filename),                     'Error: Unable to find {} to load the predictions for sample token {} (lidar '                     'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
            else:
                if len(nusc.lidarseg) > 0:  # Ensure lidarseg.json is not empty (e.g. in case of v1.0-test).
                    lidarseg_labels_filename = osp.join(NuScenes.nusc.dataroot,
                                                        NuScenes.nusc.get('lidarseg', pointsensor_token)['filename'])
                else:
                    lidarseg_labels_filename = None

            if lidarseg_labels_filename:
                # Paint each label in the pointcloud with a RGBA value.
                coloring = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                              NuScenes.nusc.lidarseg_name2idx_mapping, NuScenes.nusc.colormap)
            else:
                coloring = depths
                print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                      'from the ego vehicle instead.'.format(NuScenes.nusc.version))
        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring, im
    
lidarseg_preds_bin_path= ''    
    
def render_pointcloud_in_image(sample_token: str,
                               dot_size: int = 5,
                               pointsensor_channel: str = 'RADAR_FRONT',
                               camera_channel: str = 'CAM_FRONT',
                               out_path: str = None,
                               render_intensity: bool = False,
                               show_lidarseg: bool = False,
                               filter_lidarseg_labels: list = None,
                               ax: Axes = None,
                               show_lidarseg_legend: bool = False,
                               verbose: bool = True,
                               lidarseg_preds_bin_path: str = None):
    """
    Scatter-plots a pointcloud on top of image.
    :param sample_token: Sample token.
    :param dot_size: Scatter plot dot size.
    :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param out_path: Optional path to save the rendered figure to disk.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
    :param ax: Axes onto which to render.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param verbose: Whether to display the image in a window.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    """
    sample_record = nusc.get('sample', sample_token)

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]

    points, coloring, im = map_pointcloud_to_image(pointsensor_token, camera_token,
                                                        render_intensity=render_intensity,
                                                        show_lidarseg=show_lidarseg,
                                                        filter_lidarseg_labels=filter_lidarseg_labels,
                                                        lidarseg_preds_bin_path=lidarseg_preds_bin_path)

    # Init axes.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))
        if lidarseg_preds_bin_path:
            fig.canvas.set_window_title(sample_token + '(predictions)')
        else:
            fig.canvas.set_window_title(sample_token)
    else:  # Set title on if rendering as part of render_sample.
        ax.set_title(camera_channel)
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    ax.axis('off')

    # Produce a legend with the unique colors from the scatter.
    if pointsensor_channel == 'LIDAR_TOP' and show_lidarseg and show_lidarseg_legend:
        # Since the labels are stored as class indices, we get the RGB colors from the colormap in an array where
        # the position of the RGB color corresponds to the index of the class it represents.
        color_legend = colormap_to_colors(self.nusc.colormap, self.nusc.lidarseg_name2idx_mapping)

        # If user does not specify a filter, then set the filter to contain the classes present in the pointcloud
        # after it has been projected onto the image; this will allow displaying the legend only for classes which
        # are present in the image (instead of all the classes).
        if filter_lidarseg_labels is None:
            filter_lidarseg_labels = get_labels_in_coloring(color_legend, coloring)

        create_lidarseg_legend(filter_lidarseg_labels,
                               self.nusc.lidarseg_idx2name_mapping, self.nusc.colormap)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()

points, coloring, im = map_pointcloud_to_image('37091c75b9704e0daa829ba56dfa0906', 'e3d495d4ac534d54b321f50006683844')

render_pointcloud_in_image('ca9a282c9e77460f8360f564131a8af5')


# In[26]:


##### Question5.2. Visualizing LiDAR data projection on image and Calibration information#####


filter_lidarseg_labels = []
lidarseg_preds_bin_path= ''

def map_pointcloud_to_image(pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0,
                                render_intensity: bool = False,
                                show_lidarseg: bool = False,
                                filter_lidarseg_labels: list = None,
                                lidarseg_preds_bin_path: str = None):
    
        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            if show_lidarseg:
                assert hasattr(NuScenes.nusc, 'lidarseg'), 'Error: nuScenes-lidarseg not installed!'

                # Ensure that lidar pointcloud is from a keyframe.
                assert pointsensor['is_key_frame'],                     'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                assert not render_intensity, 'Error: Invalid options selected. You can only select either '                                              'render_intensity or show_lidarseg, not both.'

            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(nusc.dataroot, cam['filename']))

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        print(cs_record)
        
        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        if render_intensity:
            assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, '                                                               'not %s!' % pointsensor['sensor_modality']
            # Retrieve the color from the intensities.
            # Performs arbitary scaling to achieve more visually pleasing results.
            intensities = pc.points[3, :]
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities = intensities ** 0.1
            intensities = np.maximum(0, intensities - 0.5)
            coloring = intensities
        elif show_lidarseg:
            assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, '                                                               'not %s!' % pointsensor['sensor_modality']

            if lidarseg_preds_bin_path:
                sample_token = nusc.get('sample_data', pointsensor_token)['sample_token']
                lidarseg_labels_filename = lidarseg_preds_bin_path
                assert os.path.exists(lidarseg_labels_filename),                     'Error: Unable to find {} to load the predictions for sample token {} (lidar '                     'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
            else:
                if len(nusc.lidarseg) > 0:  # Ensure lidarseg.json is not empty (e.g. in case of v1.0-test).
                    lidarseg_labels_filename = osp.join(NuScenes.nusc.dataroot,
                                                        NuScenes.nusc.get('lidarseg', pointsensor_token)['filename'])
                else:
                    lidarseg_labels_filename = None

            if lidarseg_labels_filename:
                # Paint each label in the pointcloud with a RGBA value.
                coloring = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                              NuScenes.nusc.lidarseg_name2idx_mapping, NuScenes.nusc.colormap)
            else:
                coloring = depths
                print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                      'from the ego vehicle instead.'.format(NuScenes.nusc.version))
        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring, im
    
    
def render_pointcloud_in_image(sample_token: str,
                               dot_size: int = 5,
                               pointsensor_channel: str = 'LIDAR_TOP',
                               camera_channel: str = 'CAM_FRONT',
                               out_path: str = None,
                               render_intensity: bool = False,
                               show_lidarseg: bool = False,
                               filter_lidarseg_labels: List = None,
                               ax: Axes = None,
                               show_lidarseg_legend: bool = False,
                               verbose: bool = True,
                               lidarseg_preds_bin_path: str = None):
    """
    Scatter-plots a pointcloud on top of image.
    :param sample_token: Sample token.
    :param dot_size: Scatter plot dot size.
    :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param out_path: Optional path to save the rendered figure to disk.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
    :param ax: Axes onto which to render.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param verbose: Whether to display the image in a window.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    """
    sample_record = nusc.get('sample', sample_token)

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]

    points, coloring, im = map_pointcloud_to_image(pointsensor_token, camera_token,
                                                        render_intensity=render_intensity,
                                                        show_lidarseg=show_lidarseg,
                                                        filter_lidarseg_labels=filter_lidarseg_labels,
                                                        lidarseg_preds_bin_path=lidarseg_preds_bin_path)

    # Init axes.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))
        if lidarseg_preds_bin_path:
            fig.canvas.set_window_title(sample_token + '(predictions)')
        else:
            fig.canvas.set_window_title(sample_token)
    else:  # Set title on if rendering as part of render_sample.
        ax.set_title(camera_channel)
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    ax.axis('off')

    # Produce a legend with the unique colors from the scatter.
    if pointsensor_channel == 'LIDAR_TOP' and show_lidarseg and show_lidarseg_legend:
        # Since the labels are stored as class indices, we get the RGB colors from the colormap in an array where
        # the position of the RGB color corresponds to the index of the class it represents.
        color_legend = colormap_to_colors(self.nusc.colormap, self.nusc.lidarseg_name2idx_mapping)

        # If user does not specify a filter, then set the filter to contain the classes present in the pointcloud
        # after it has been projected onto the image; this will allow displaying the legend only for classes which
        # are present in the image (instead of all the classes).
        if filter_lidarseg_labels is None:
            filter_lidarseg_labels = get_labels_in_coloring(color_legend, coloring)

        create_lidarseg_legend(filter_lidarseg_labels,
                               self.nusc.lidarseg_idx2name_mapping, self.nusc.colormap)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()

points, coloring, im = map_pointcloud_to_image('9d9bf11fb0e144c8b446d54a8a00184f', 'e3d495d4ac534d54b321f50006683844')

render_pointcloud_in_image('ca9a282c9e77460f8360f564131a8af5')

