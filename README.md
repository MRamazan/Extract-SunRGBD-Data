Extracts: <br>
image: 000001.jpg,...<br>
depth: 000001.png,...<br>


label:[classname, centroid1,centroid2,centroid3,w,l,h,orientation1,orientatio2]


```bash 
#pcd: [N,6] point cloud data created from depth image <br>
#np.array([N, 6]) N = number of points <br>
pcd = np.load("pcd/000001.npz") <br>
points = pcd[:, 0:3] <br>
colors = pcd[:, 3:6]
```

calib: {Rtilt: [[x,x,x],[x,x,x],[x,x,x]], K:[[fx,0,cx],[0,fy,cy],[0,0,1]]}    <br>    
```bash 
with open("calib/000001.txt", "r") as file:
     calib = json.load(file)
     Rtilt = calib["Rtilt"]
     K = calib["K"]
#np.load("votes/000001.npz")
```
<br><br>

VISUALIZE PCD:<br>
![](example_imgs/pcd.png)<br>
VISUALIZE 3D BBOXES ON 2D IMAGE:<br>
![](example_imgs/3dbboxes_on_2d_image.png)

