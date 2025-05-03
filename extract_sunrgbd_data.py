import argparse
import json
import os
import cv2
import numpy as np
import scipy.io as sio
from depth_to_pcd import ExampleDepthPointCloud
import sun_utils

classes = ['bookshelf', 'chair', 'sofa', 'table', 'computer', 'desk', 'keyboard', 'paper',
           'garbage_bin', 'monitor', 'sofa_chair', 'box', 'recycle_bin', 'cpu', 'whiteboard',
           'shelf', 'endtable', 'cabinet', 'lamp', 'drawer', 'painting', 'sink', 'picture',
           'coffee_table', 'night_stand', 'bed', 'toilet', 'pillow', 'dresser', 'stool']

def extract_sunrgbd_data(save_folder, save=False, save_imgs=False, save_pcd=False, save_votes=False,sample_point_count=55000):
    depth_to_pcd = ExampleDepthPointCloud()
    missing_annotation = 0
    no_obj = 0
    img_paths = []

    os.makedirs(save_folder, exist_ok=True)
    for subdir in ["label", "calib", "image", "depth", "pcd", "votes"]:
        os.makedirs(os.path.join(save_folder, subdir), exist_ok=True)

    matlab_meta = sio.loadmat("SUNRGBDMeta3DBB_v2.mat")
    for i in range(len(matlab_meta['SUNRGBDMeta'][0])):
        if i - 1 == 10335 - missing_annotation:
            break
        file_idx = i - no_obj + 1
        while len(matlab_meta["SUNRGBDMeta"][0][i][10]) == 0:
            missing_annotation += 1
            i = i + 1
        print(i)

        camera_calib = {}
        label_filename = os.path.join(save_folder, "label", f'{file_idx:06d}.txt')
        calib_filename = os.path.join(save_folder, "calib", f'{file_idx:06d}.txt')
        image_filename = os.path.join(save_folder, "image", f'{file_idx:06d}')
        depth_filename = os.path.join(save_folder, "depth", f'{file_idx:06d}')
        pcd_filename = os.path.join(save_folder, "pcd", f'{file_idx:06d}')
        votes_filename = os.path.join(save_folder, "votes", f'{file_idx:06d}')

        rgb_img_dir = "metadata/" + matlab_meta["SUNRGBDMeta"][0][i][4][0].replace("//", "/")[17:]
        depth_img_dir = "metadata/" + matlab_meta["SUNRGBDMeta"][0][i][3][0].replace("//", "/")[17:].replace("depth", "depth_bfx")
        camera_calib["Rtilt"] = matlab_meta["SUNRGBDMeta"][0][i][1]
        camera_calib["K"] = matlab_meta["SUNRGBDMeta"][0][i][2]
        all_objs = []

        for obj_idx in range(len(matlab_meta["SUNRGBDMeta"][0][i][10][0])):
            label = matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][3][0]
            if label in classes:
                width, length, height = (matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][1][0][idx] for idx in range(3))
                w, l, h = round(np.float32(width), 6), round(np.float32(length), 6), round(np.float32(height), 6)
                centroid = matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][2][0]
                c1, c2, c3 = map(lambda x: round(np.float32(x), 6), centroid[:3])
                o1 = round(np.float32(matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][5][0][0]), 6)
                o2 = round(np.float32(matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][5][0][1]), 6)

                all_objs.append([label, c1, c2, c3, w, l, h, o1, o2])

        if len(all_objs) == 0:
            no_obj += 1
            continue
        else:
            img_paths.append(rgb_img_dir)

        if save:
            with open(label_filename, "w") as f:
                for obj in all_objs:
                    f.write(" ".join(map(str, obj)) + "\n")
            with open("img_paths.txt", "w") as f:
                for path in img_paths:
                    f.write(os.path.dirname(os.path.dirname(path)) + '\n')
            with open(calib_filename, "w") as f:
                json.dump({k: v.tolist() for k, v in camera_calib.items()}, f)

        if save_imgs:
            image = cv2.imread(rgb_img_dir)
            depth = cv2.imread(depth_img_dir, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(image_filename + ".jpg", image)
            cv2.imwrite(depth_filename + ".png", depth)

        if save_pcd:
            image = cv2.imread(rgb_img_dir)
            depth = cv2.imread(depth_img_dir, cv2.IMREAD_GRAYSCALE)
            pcd, pcd_color = depth_to_pcd.run(image, depth, camera_calib["Rtilt"], camera_calib["K"])
            stacked = np.concatenate((pcd, pcd_color), axis=1)
            sampled = sun_utils.random_sampling(stacked, sample_point_count)
            np.savez_compressed(pcd_filename, pc=sampled)
            if save_votes:
                N = sampled.shape[0]
                point_votes = np.zeros((N, 10))
                point_vote_idx = np.zeros(N, dtype=np.int32)
                indices = np.arange(N)
                objects = sun_utils.read_sunrgbd_label(all_objs)
                for obj in objects:
                    if obj.classname not in classes: continue
                    try:
                        box3d_pts = sun_utils.my_compute_box_3d(obj.centroid,
                                                                np.array([obj.l, obj.w, obj.h]),
                                                                obj.heading_angle)
                        pc_in_box3d, inds = sun_utils.extract_pc_in_box3d(sampled, box3d_pts)
                        point_votes[inds, 0] = 1
                        votes = np.expand_dims(obj.centroid, 0) - pc_in_box3d[:, 0:3]
                        sparse_inds = indices[inds]
                        for i, j in enumerate(sparse_inds):
                            point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i]
                            if point_vote_idx[j] == 0:
                                point_votes[j, 4:7] = votes[i]
                                point_votes[j, 7:10] = votes[i]
                        point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)
                    except:
                        print("ERROR ----", file_idx, obj.classname)
                np.savez_compressed(votes_filename, point_votes=point_votes)



'''

save=True : saves labels [class,centroid1,centroid2,centroid3,w,l,h,orientation1,orientation2] , label/000001.txt

save_imgs=True : saves images as jpg and depth images as png [image/0000001.jpg] depth/0000001.png]

save_pcd=True : saves point cloud data created from depth image as .npz file, 
                np.array([N, 6]) N = number of points, points = pcd[:, 0:3] , colors = pcd[:, 3:6] (it takes_time)

save_votes=True : saves votes as .npz file

'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SUNRGBD data with options.")
    parser.add_argument("--save_folder", type=str, default="sunrgbd_labels", help="Folder to save output.")
    parser.add_argument("--save", default=True, help="Save labels and calibration.")
    parser.add_argument("--save_imgs", default=True, help="Save RGB and depth images.")
    parser.add_argument("--save_pcd", default=True, help="Save point cloud data.")
    parser.add_argument("--save_votes", default=False, help="Save voting data.")
    parser.add_argument("--sample_point_count", default=100000, help="sample point count")

    args = parser.parse_args()

    extract_sunrgbd_data(
        save_folder=args.save_folder,
        save=args.save,
        save_imgs=args.save_imgs,
        save_pcd=args.save_pcd,
        save_votes=args.save_votes
    )

