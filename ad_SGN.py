import copy
import time
import random
import argparse
import numpy as np
import scipy.sparse as sp

from utils.io import load_tiff_pc
from probreg.cpd import AffineCPD
from sklearn.neighbors import KDTree
from utils.graphics import plot_pointcloud
from manifold.utils import adjacency_matrix
from sklearn.metrics.pairwise import cosine_distances
from manifold.coupling import couple_adjacency_matrices
from manifold.spectral_embedding import spectral_embedding
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from utils.pre_processing import largest_connected_component, ransac_registration, preprocess

_SEED = 42
np.random.seed = _SEED
random.seed = _SEED


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--cross', type=float, default=1,
                        help='Fraction of source nodes used for coupled graph building')
    parser.add_argument('-p', '--points', type=int, default=13000,
                        help='Number of points to subsample for matching')
    parser.add_argument('-th', '--threshold', type=float, default=0.001,
                        help='Threshold for plane removal')
    parser.add_argument('-v', '--voxel_size', type=float, default=0.001,
                        help='RANSAC voxel size')
    parser.add_argument('-d', '--cpd_down', type=int, default=3000,
                        help='Downsampling to speed-up CPD')
    parser.add_argument('-s', '--source', type=str, default='./data/cable_gland/train/good/xyz/000.tiff',
                        help='path to source shape')
    parser.add_argument('-t', '--target', type=str, default='./data/cable_gland/test/bent/xyz/020.tiff',
                        help='path to target shape')
    parser.add_argument('-k', '--k', type=int, default=5,
                        help='iterations of the Simple Graph Convolution')
    args = parser.parse_args()

    start = time.time()
    # Load source point cloud
    source, source_pca, source_sampling = load_tiff_pc(
        args.source,
        n_points=args.points,
        threshold=args.threshold
    )
    target, target_pca, target_sampling = load_tiff_pc(
        args.target,
        n_points=args.points,
        threshold=args.threshold
    )

    source = source.astype("float64")
    target = target.astype("float64")

    adjacency_source = adjacency_matrix(source, mode="gaussian", method="knn")
    adjacency_target = adjacency_matrix(target, mode="gaussian", method="knn")
    source = largest_connected_component(source, adj=adjacency_source, num_components=1)
    target = largest_connected_component(target, adj=adjacency_target, num_components=1)

    # print("loaded source and target adjacency matrices")
    # We do first RANSAC registration and then Affine CPD since the latter may fail if point clouds
    # are initially very far
    source = ransac_registration(
        source=source,
        target=target,
        voxel_size=args.voxel_size
    )

    # Rough Affine CPD on downsampled point clouds in order to have an idea of the transformation and
    # do it faster
    # print("Registration completed")
    print("Affine CPD on sub-sampled point cloud (", args.cpd_down, "points).")
    source_ds = source[np.random.choice(range(len(source)), args.cpd_down, replace=False)]
    target_ds = target[np.random.choice(range(len(target)), args.cpd_down, replace=False)]
    affine_cpd = AffineCPD(source=source_ds)
    reg = affine_cpd.registration(target_ds)
    source = reg.transformation.transform(source)
    source_subem = copy.deepcopy(source)
    target_subem = copy.deepcopy(target)
    # Re-compute adjacencies and degree matrices because they may have changed with pre-processing
    for i in range(args.k):
        adjacency_source = adjacency_matrix(source_subem, mode="gaussian", method="knn")
        _, dd_source = csgraph_laplacian(
            adjacency_source, normed=False, return_diag=True
        )
        adjacency_target = adjacency_matrix(target_subem, mode="gaussian", method="knn")
        _, dd_target = csgraph_laplacian(
            adjacency_target, normed=False, return_diag=True
        )
        # Stochastically select target points for cross connections
        n_cross = int(args.cross * len(target))
        cross_target = np.random.choice(
            range(len(target)),
            size=n_cross,
            replace=False
        )
        # if l=0, we need empty lists to not have errors later
        if n_cross > 0:
            cross_kdtree = KDTree(source)
            dist, cross_source = cross_kdtree.query(target[cross_target])
            dist = np.exp(-dist ** 2 / np.max(dist) ** 2)
        else:
            dist = np.array([])
            cross_source = np.array([])
        print("Coupling graphs ...")
        print(":: Number of cross connections", n_cross)
        # Compute coupled adjacency
        adjacency_coupled = couple_adjacency_matrices(
            adjacency_a=adjacency_source+np.identity(adjacency_source.shape[0]),
            adjacency_b=adjacency_target+np.identity(adjacency_target.shape[0]),
            inds_a=cross_source.reshape(-1),
            inds_b=cross_target.reshape(-1),
            dist=sp.diags(dist.reshape(-1)),
            return_uncoupled=False
        )
        coupled_laplacian = csgraph_laplacian(adjacency_coupled, normed=True)
        coupled_s = np.identity((coupled_laplacian.shape)[0]) - coupled_laplacian
        coupled_maps = np.vstack((source, target))
        coupled_embeddings = coupled_s * coupled_maps
        # Split them
        source_embeddings = coupled_embeddings[:len(source)]
        target_embeddings = coupled_embeddings[len(source):]
        # Embedding restriceted to cross nodes
        source_subemb = source_embeddings[cross_source.reshape(-1), :]
        target_subemb = target_embeddings[cross_target.reshape(-1), :]
    # point to point distance
    p2p_dist = cosine_distances(np.asarray(source_subemb), np.asarray(target_subemb)).diagonal()
    print("Anomaly localization performed in ", time.time()-start, "seconds.")
    # Plot prediction
    fig = plot_pointcloud(
        target[cross_target],
        marker=dict(
            size=3,
            color=p2p_dist,
            colorscale="Jet",
            cmax=1,
            showscale=True
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
    fig.show()
