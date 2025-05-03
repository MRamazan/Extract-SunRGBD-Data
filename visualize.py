import numpy as np
import open3d as o3d
import argparse
from sun_utils import Visualize

def main():
    # Argparse ile komut satırı argümanlarını al
    parser = argparse.ArgumentParser(description="Visualize 3D bounding boxes and point cloud data.")
    parser.add_argument("--image_index", type=int, help="Index of the image to visualize")
    parser.add_argument("--visualize_function", type=int, choices=[1, 2], help="Select visualization function: 1 for 3D BBoxes on 2D Image, 2 for Point Cloud and BBoxes")

    args = parser.parse_args()

    # Visualize nesnesi
    visualize = Visualize()

    # Görselleri ve kalibrasyonu al
    rgb_image = visualize.get_rgb_image(args.image_index)
    depth = visualize.get_depth_image(args.image_index)
    calib = visualize.get_calib(args.image_index)
    Rtilt = calib["Rtilt"]
    K = calib["K"]

    # PCD ve 3D köşeleri al
    pcd = visualize.get_pcd(args.image_index)
    objects = visualize.get_label_objects(args.image_index)
    corners3d = visualize.get_corners_3d(objects)

    # Fonksiyonlar
    def visualize_3d_bboxes_on_2d_image(rgb_image, corners3d, K, Rtilt):
        visualize.visualize_3d_bboxes_on_2d_image(rgb_image, corners3d, K, Rtilt)

    def visualize_point_cloud_data_and_bboxes(pcd, corners3d):
        points = pcd[:, 0:3]
        color = pcd[:, 3:6]
        point_cloud = o3d.geometry.PointCloud()

        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(color))
        visualize.visualize_point_cloud_data_and_bboxes(point_cloud, corners3d)

    # Seçilen görselleştirme fonksiyonuna göre işlem yap
    if args.visualize_function == 1:
        print("Visualizing 3D BBoxes on 2D image...")
        visualize_3d_bboxes_on_2d_image(rgb_image, corners3d, K, Rtilt)
    elif args.visualize_function == 2:
        print("Visualizing Point Cloud and BBoxes...")
        visualize_point_cloud_data_and_bboxes(pcd, corners3d)

if __name__ == "__main__":
    main()
