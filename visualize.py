from sun_utils import Visualize
import numpy as np
import open3d as o3d
image_index = 1 #Dont set this to 0 because there is no file named 000000.txt
visualize = Visualize()

rgb_image = visualize.get_rgb_image(image_index)
depth = visualize.get_depth_image(image_index)
calib = visualize.get_calib(image_index)
Rtilt = calib["Rtilt"]
K = calib["K"]

pcd = visualize.get_pcd(image_index)
objects = visualize.get_label_objects(image_index)
corners3d = visualize.get_corners_3d(objects)

def VISUALIZE_3D_BBOXES_ON_2D_IMAGE(rgb_image,corners3d, K, Rtilt):
    visualize.visualize_3d_bboxes_on_2d_image(rgb_image, corners3d, K, Rtilt)

def VISUALIZE_POINT_CLOUD_DATA_AND_BBOXES(pcd, corners3d):
    points = pcd[:, 0:3]
    color = pcd[:, 3:6]
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(color))
    visualize.visualize_point_cloud_data_and_bboxes(point_cloud, corners3d)

'''VISUALIZE POINT CLOUD DATA AND BBOXES'''
#VISUALIZE_POINT_CLOUD_DATA_AND_BBOXES(pcd, corners3d)

'''VISUALIZE 3D BBOXES ON 2D IMAGE'''
#VISUALIZE_3D_BBOXES_ON_2D_IMAGE(rgb_image, corners3d, K, Rtilt)
































