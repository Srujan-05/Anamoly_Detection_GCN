import open3d as o3d
import numpy as np
from utils.pre_processing import ransac_registration

randomArr = np.random.rand(30000, 3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(randomArr)

randomArr = ransac_registration(randomArr, randomArr, 0.001)


