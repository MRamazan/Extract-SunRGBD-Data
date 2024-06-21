import json
import os

import cv2
import numpy as np
import open3d as o3d

class SUNObject3d(object):
    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.centroid = np.array([data[1],data[2],data[3]])
        self.unused_dimension = np.array([data[4],data[5],data[6]])
        self.w = data[4]
        self.l = data[5]
        self.h = data[6]
        self.orientation = np.zeros((3,))
        self.orientation[0] = data[7]
        self.orientation[1] = data[8]
        self.heading_angle = -1 * np.arctan2(self.orientation[1], self.orientation[0])

class SUNObject3d_v2(object):
    def __init__(self, data):
        data = data
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.centroid = np.array([data[1],data[2],data[3]])
        self.unused_dimension = np.array([data[4],data[5],data[6]])
        self.w = data[4]
        self.l = data[5]
        self.h = data[6]
        self.orientation = np.zeros((3,))
        self.orientation[0] = data[7]
        self.orientation[1] = data[8]
        self.heading_angle = -1 * np.arctan2(self.orientation[1], self.orientation[0])

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """Input is NxC, output is num_samplexC"""
    if replace is None:
        replace = pc.shape[0] < num_sample
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1*heading_angle)
    l,w,h = size
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


def read_sunrgbd_label(objects):
    objects = [SUNObject3d_v2(obj) for obj in objects]
    return objects


class Visualize():
    def __init__(self):
        self.save_file = "sunrgbd_labels"
        self.label_dir = os.path.join(self.save_file, "label")
        self.depth_dir = os.path.join(self.save_file, "depth")
        self.image_dir = os.path.join(self.save_file, "image")
        self.pcd_dir = os.path.join(self.save_file, "pcd")
        self.calib_dir = os.path.join(self.save_file, "calib")
        self.classes = ['bookshelf', 'chair', 'sofa', 'table', 'computer', 'desk', 'keyboard', 'paper',
                       'garbage_bin', 'monitor', 'sofa_chair', 'box', 'recycle_bin', 'cpu', 'whiteboard',
                       'shelf', 'endtable', 'cabinet', 'lamp', 'drawer', 'painting', 'sink', 'picture',
                       'coffee_table', 'night_stand', 'bed', 'toilet', 'pillow', 'dresser', 'stool']

    def get_corners_3d(self, objects):
        object_list = []
        corners_3d_pc = []
        for obj in objects:
            if obj.classname in self.classes:
             obb = np.zeros((8))
             obb[0:3] = obj.centroid
             obb[3:6] = np.array([obj.l, obj.w, obj.h])
             obb[6] = obj.heading_angle
             object_list.append(obb)
        bboxes = np.vstack(object_list)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = self.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            corners_3d_pc.append(corners_3d)

        return corners_3d_pc

    def get_pcd(self, idx):
        pcd_filename = os.path.join(self.pcd_dir, '%06d.npz' % (idx))
        pcd = np.load(pcd_filename)["pc"]
        return pcd

    def read_sunrgbd_label(self, label_filename):
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [SUNObject3d(line) for line in lines]
        return objects

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        print(label_filename)
        return self.read_sunrgbd_label(label_filename)

    def get_rgb_image(self, idx):
        rgb_filename = os.path.join(self.image_dir, '%06d.jpg' % (idx))
        rgb_image = cv2.imread(rgb_filename)
        return rgb_image

    def get_depth_image(self, idx):
        depth_filename = os.path.join(self.depth_dir, '%06d.png' % (idx))
        depth = cv2.imread(depth_filename, cv2.IMREAD_GRAYSCALE)
        return depth

    def get_calib(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        with open(calib_filename, "r") as file:
            calib = json.load(file)
        return calib

    def rotz(self,t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    def visualize_3d_bboxes_on_2d_image(self, image, point_cloud_data, K, Rtilt):
        if point_cloud_data:
            for pcd in point_cloud_data:
                corners_2d, _ = self.project_upright_depth_to_image(pcd, K, Rtilt)
                self.draw_projected_box3d(image, corners_2d)

            cv2.imshow("image", image)
            cv2.waitKey(0)

    def project_upright_depth_to_image(self, pc, K, Rtilt):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc2 = self.project_upright_depth_to_camera(Rtilt, pc)
        uv = np.dot(pc2, np.transpose(K))  # (n,3)
        uv[:, 0] /= uv[:, 2]
        uv[:, 1] /= uv[:, 2]
        return uv[:, 0:2], pc2[:, 2]

    def flip_axis_to_camera(self, pc):
        ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
                Input and output are both (N,3) array
        '''
        pc2 = np.copy(pc)
        pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
        pc2[:, 1] *= -1
        return pc2

    def project_upright_depth_to_camera(self, Rtilt, pc):
        ''' project point cloud from depth coord to camera coordinate
                Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(Rtilt), np.transpose(pc[:, 0:3]))  # (3,n)
        return self.flip_axis_to_camera(np.transpose(pc2))

    def my_compute_box_3d(self,center, size, heading_angle):
        R = self.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)

    def draw_projected_box3d(self, image, qs, color=(255, 255, 255), thickness=2):
        ''' Draw 3d bounding box in image
            qs: (8,2) array of vertices for the 3d box in following order:
                1 -------- 0
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7
        '''
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness,
                     cv2.LINE_AA)  # use LINE_AA for opencv3

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
        return image

    def visualize_point_cloud_data_and_bboxes(self, pcd, bboxes):
        pcd_list = []
        pcd_list.append(pcd)
        for box in bboxes:
         bbox_points = o3d.geometry.PointCloud()
         bbox_points.points = o3d.utility.Vector3dVector(box)
         bbox = o3d.geometry.OrientedBoundingBox.create_from_points(bbox_points.points)
         bbox.color = (1, 0, 0)  # Kırmızı renkte bounding box
         pcd_list.append(bbox_points)
         pcd_list.append(bbox)



        o3d.visualization.draw_geometries(pcd_list)
