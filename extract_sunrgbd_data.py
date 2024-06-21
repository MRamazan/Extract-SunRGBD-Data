import json
import os
import cv2
import numpy as np
import scipy.io as sio
from depth_to_pcd import ExampleDepthPointCloud
import sun_utils


'''

save=True : saves labels [class,centroid1,centroid2,centroid3,w,l,h,orientation1,orientation2] , label/000001.txt

save_imgs=True : saves images as jpg and depth images as png [image/0000001.jpg] depth/0000001.png]

save_pcd=True : saves point cloud data created from depth image as .npz file, 
                np.array([N, 6]) N = number of points, points = pcd[:, 0:3] , colors = pcd[:, 3:6]

save_votes=True : saves votes as .npz file

'''


save_folder = "sunrgbd_labels"
classes =['bookshelf', 'chair', 'sofa', 'table', 'computer', 'desk', 'keyboard', 'paper',
                       'garbage_bin', 'monitor', 'sofa_chair', 'box', 'recycle_bin', 'cpu', 'whiteboard',
                       'shelf', 'endtable', 'cabinet', 'lamp', 'drawer', 'painting', 'sink', 'picture',
                       'coffee_table', 'night_stand', 'bed', 'toilet', 'pillow', 'dresser', 'stool']
def extract_sunrgbd_data(save_folder,save=False, save_imgs=False, save_pcd=False, save_votes=False):
    depth_to_pcd = ExampleDepthPointCloud()
    missing_annotation = 0
    no_obj = 0
    img_paths = []


    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(os.path.join(save_folder , "label")):
        os.mkdir(os.path.join(save_folder , "label"))
    if not os.path.exists(os.path.join(save_folder , "calib")):
        os.mkdir(os.path.join(save_folder , "calib"))
    if not os.path.exists(os.path.join(save_folder , "image")):
        os.mkdir(os.path.join(save_folder , "image"))
    if not os.path.exists(os.path.join(save_folder , "depth")):
        os.mkdir(os.path.join(save_folder , "depth"))
    if not os.path.exists(os.path.join(save_folder , "pcd")):
        os.mkdir(os.path.join(save_folder , "pcd"))
    if not os.path.exists(os.path.join(save_folder , "votes")):
        os.mkdir(os.path.join(save_folder , "votes"))
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

        label_filename = os.path.join(save_folder,"label", '%06d.txt' % (file_idx))
        calib_filename = os.path.join(save_folder, "calib", '%06d.txt' % (file_idx))
        image_filename = os.path.join(save_folder, "image", '%06d' % (file_idx))
        depth_filename = os.path.join(save_folder, "depth", '%06d' % (file_idx))
        pcd_filename = os.path.join(save_folder, "pcd", '%06d' % (file_idx))
        votes_filename = os.path.join(save_folder, "votes", '%06d' % (file_idx))

        #image_folder = matlab_meta["SUNRGBDMeta"][0][i][0][0]
        rgb_img_dir = "metadata/" + matlab_meta["SUNRGBDMeta"][0][i][4][0].replace("//", "/")[17:]
        depth_img_dir = "metadata/" + matlab_meta["SUNRGBDMeta"][0][i][3][0].replace("//", "/")[17:].replace("depth", "depth_bfx")
        camera_calib["Rtilt"] = matlab_meta["SUNRGBDMeta"][0][i][1]
        camera_calib["K"] = matlab_meta["SUNRGBDMeta"][0][i][2]
        all_objs = []


        for obj_idx in range(0,len(matlab_meta["SUNRGBDMeta"][0][i][10][0])):
           label = matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][3][0]
           if label in classes:
              width,length,height = (matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][1][0][idx] for idx in range(0,3))
              w,l,h = round(np.float32(width),6), round(np.float32(length),6), round(np.float32(height),6)
              centroid = matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][2][0]
              c_1, c_2, c_3 = round(np.float32(centroid[0]), 6), round(np.float32(centroid[1]), 6), round(np.float32(centroid[2]), 6)
              orientation_1 = round(np.float32(matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][5][0][0]),6)
              orientation_2 = round(np.float32(matlab_meta["SUNRGBDMeta"][0][i][10][0][obj_idx][5][0][1]),6)

              obj_infos = [label,c_1, c_2, c_3, w, l, h, orientation_1, orientation_2]
              all_objs.append(obj_infos)
           else:
               continue
        if len(all_objs) == 0:
            no_obj += 1
        else:
            img_paths.append(rgb_img_dir)


        if save:
          with open(label_filename, "w") as file:
            for obj in all_objs:
              for info in obj:
                file.write(str(info))
                file.write(" ")
              file.write("\n")



          file_name = "img_paths.txt"
          with open(file_name, 'w', encoding='utf-8') as file:
              for item in img_paths:
                  item = os.path.dirname(item)
                  item = os.path.dirname(item)
                  file.write(item + '\n')



          with open(calib_filename, "w") as file:
              camera_calib_converted = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in camera_calib.items()}
              json.dump(camera_calib_converted, file)


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
          sampled = sun_utils.random_sampling(stacked, 175000)
          np.savez_compressed(pcd_filename, pc=sampled)
          if save_votes:
              N = sampled.shape[0]
              point_votes = np.zeros((N, 10))
              point_vote_idx = np.zeros((N)).astype(np.int32)
              indices = np.arange(N)
              objects = sun_utils.read_sunrgbd_label(all_objs)
              for obj in objects:
                  if obj.classname not in classes: continue
                  try:

                      # Find all points in this object's OBB
                      box3d_pts_3d = sun_utils.my_compute_box_3d(obj.centroid,
                                                                     np.array([obj.l, obj.w, obj.h]), obj.heading_angle)
                      pc_in_box3d, inds = sun_utils.extract_pc_in_box3d(sampled, box3d_pts_3d)
                      # Assign first dimension to indicate it is in an object box
                      point_votes[inds, 0] = 1
                      # Add the votes (all 0 if the point is not in any object's OBB)
                      votes = np.expand_dims(obj.centroid, 0) - pc_in_box3d[:, 0:3]
                      sparse_inds = indices[inds]  # turn dense True,False inds to sparse number-wise inds
                      for i in range(len(sparse_inds)):
                          j = sparse_inds[i]
                          point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i,
                                                                                                                :]
                          # Populate votes with the fisrt vote
                          if point_vote_idx[j] == 0:
                              point_votes[j, 4:7] = votes[i, :]
                              point_votes[j, 7:10] = votes[i, :]
                      point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)
                  except:
                      print('ERROR ----', file_idx, obj.classname)

              np.savez_compressed(votes_filename, point_votes=point_votes)






'''

save=True : saves labels [class,centroid1,centroid2,centroid3,w,l,h,orientation1,orientation2]

save_imgs=True : saves images as jpg and depth images as png [image/0000001.jpg] depth/0000001.png]

save_pcd=True : saves point cloud data created from depth image, np.array([N, 6]) N = number of points, points = pcd[:, 0:3] , colors = pcd[:, 3:6]

save_votes=True : saves votes 

'''


extract_sunrgbd_data(save_folder, True, True, True, True)
