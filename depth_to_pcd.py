import cv2
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import yaml

from PIL import Image

from mpl_toolkits.mplot3d import Axes3D

class ExampleDepthPointCloud:
    def __init__(self):
      pass



    def load_image(self, name):
        img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        return np.array(img)



    def depth_to_3d(self,rgb, depth, K):
        cloud = []
        cloud_color = []
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]


        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                x = (j - cx) * depth[i, j] / fx
                y = (i - cy) * depth[i, j] / fy
                z = depth[i, j]

                cloud.append([x, z, -y])
                cloud_color.append(rgb[i, j].tolist())
        return np.array(cloud), np.array(cloud_color)

    def visualize_point_cloud(self, cloud, cloud_color):
        # Visualize point cloud using matplotlib
        print(cloud)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(cloud)
        pcd2.colors = o3d.utility.Vector3dVector(cloud_color / 255)
        o3d.visualization.draw_geometries([pcd2])


    def visualize_depth_image(self, depth):
        # Visualize depth image using matplotlib
        plt.imshow(depth, cmap='gray')
        plt.title('Depth Image')
        plt.show()

    def run(self, rgb_img, depth_img, Rtilt, K):


        cloud, cloud_color = self.depth_to_3d(rgb_img, depth_img, K)
        cloud = np.dot(Rtilt,np.array(cloud).T).T / 31
        #self.visualize_point_cloud(cloud, cloud_color)
        #self.visualize_depth_image(depth)
        return cloud , cloud_color / 255


if __name__ == "__main__":

    example = ExampleDepthPointCloud()
    example.run()




