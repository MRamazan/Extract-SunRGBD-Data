Extracts: 


label:[classname, centroid1,centroid2,centroid3,w,l,h,orientation1,orientatio2]


pcd: [N,6] point cloud data created from depth image,
     np.array([N, 6]) N = number of points, 
     pcd = np.load("000001.npz"),
     points = pcd[:, 0:3] , colors = pcd[:, 3:6]



calib: {Rtilt: [[0,0,0],[0,0,0],[0,0,0]], K:[[fx,0,cx],[0,fy,cy],[0,0,1]]}
