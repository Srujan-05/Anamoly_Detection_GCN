import open3d as o3d
import numpy as np
from scipy.sparse.csgraph import laplacian as cs_laplacian
from scipy import sparse
from utils.pre_processing import ransac_registration

adj_matrix = np.array([[0, 4], [4, 0]])

d = np.array([[0.5, 0], [0, 0.5]])
print(d*adj_matrix)
# adj_matrix = sparse.csr_matrix(adj_matrix)

laplacian = cs_laplacian(adj_matrix, normed=True)
print(laplacian)

