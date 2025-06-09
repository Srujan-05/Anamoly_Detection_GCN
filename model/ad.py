import copy

import numpy as np
import scipy.sparse as sp

from probreg.cpd import AffineCPD
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

from utils.pre_processing import ransac_registration
from manifold.coupling import couple_adjacency_matrices
from manifold.spectral_embedding import spectral_embedding
from manifold.utils import adjacency_matrix
_METHODS = [
    'spectral',
    'SGN',  # Simple Graph Convolution Network
    'GCN'  # Graph Convolution Network
]
_SEED = 42

ReluActivation = lambda x: max(0, x)
ReluActivation = np.vectorize(ReluActivation)


class AnomalyDetection:
    def __init__(
            self,
            source: np.array,
            method: str = 'spectral',
            voxel_size: int = 0.001,
            cross: float = 1,
            maps: int = 200,
            cpd_down: int = 3000,
            bins: int = 200,
            k: int = 1
    ):
        self.source = source
        if method not in _METHODS:
            raise ValueError(f"'mode' must be one of {_METHODS}, received {repr(method)}.")
        self.method = method
        self.voxel_size = voxel_size
        self.cross = cross
        self.maps = maps
        self.cpd_down = cpd_down
        self.bins = bins
        self.k = k

    def predict(
            self,
            target: np.array,
    ):
        # We do first RANSAC registration and then Affine CPD since the latter may fail if point clouds
        # are initially very far
        source = self.source.astype('float64')
        target = target.astype("float64")
        source = ransac_registration(
            source=source,
            target=target,
            voxel_size=self.voxel_size,
            verbose=False
        )
        # Rough Affine CPD on downsampled point clouds in order to have an idea of the transformation and
        # do it faster
        source_ds = source[np.random.choice(range(len(source)), self.cpd_down, replace=False)]
        target_ds = target[np.random.choice(range(len(target)), self.cpd_down, replace=False)]
        affine_cpd = AffineCPD(source=source_ds)
        reg = affine_cpd.registration(target_ds)
        source = reg.transformation.transform(source)
        # Search points for cross-connections
        n_cross = int(self.cross * len(target))
        cross_target = np.random.choice(
            range(len(target)),
            size=n_cross,
            replace=False
        )
        # Find correspondance in source
        # if l=0, we need empty lists to not have errors later
        if n_cross > 0:
            cross_kdtree = KDTree(source)
            dist, cross_source = cross_kdtree.query(target[cross_target])
            dist = np.exp(-dist ** 2 / np.max(dist) ** 2)
        else:
            dist = np.array([])
            cross_source = np.array([])
        # Now do the prediction with the desired method
        if self.method == "spectral":
            pred = self._predict_spectral(source, target, cross_source, cross_target, dist)
        elif self.method == "SGN":
            pred, mean = self._predict_sgn(source, target, cross_source, cross_target)
        else:
            pred, mean = self._predict_cgn(source, target, cross_source, cross_target)

        # We need to return the cross_target points since they might be needed to
        # create the prediction image
        return pred, cross_target.reshape(-1), mean

    def _predict_spectral(
            self,
            source,
            target,
            cross_source,
            cross_target,
            dist
    ):
        # Re-compute adjacencies and degree matrices because they may have changed with pre-processing
        adjacency_source = adjacency_matrix(source, mode="gaussian", method="knn")
        _, dd_source = csgraph_laplacian(
            adjacency_source, normed=False, return_diag=True
        )
        adjacency_target = adjacency_matrix(target, mode="gaussian", method="knn")
        _, dd_target = csgraph_laplacian(
            adjacency_target, normed=False, return_diag=True
        )
        # Compute coupled adjacency
        adjacency_coupled = couple_adjacency_matrices(
            adjacency_a=adjacency_source,
            adjacency_b=adjacency_target,
            inds_a=cross_source.reshape(-1),
            inds_b=cross_target.reshape(-1),
            dist=sp.diags(dist.reshape(-1)),
            return_uncoupled=False
        )
        # Compute coupled embeddings
        coupled_embeddings = spectral_embedding(
            adjacency=adjacency_coupled,
            n_components=self.maps,
            eigen_solver='amg',
            random_state=_SEED,
            eigen_tol='auto',
            norm_laplacian=False,
            drop_first=True,
            B=sp.diags(np.hstack(
                (dd_source,
                 dd_target
                 )
            ))
        )
        # Split them
        source_embeddings = coupled_embeddings[:len(source)]
        target_embeddings = coupled_embeddings[len(source):]
        # Embedding restriceted to cross nodes
        source_subemb = source_embeddings[cross_source.reshape(-1), :]
        target_subemb = target_embeddings[cross_target.reshape(-1), :]
        # point to point distance
        p2p_dist = cosine_distances(source_subemb, target_subemb).diagonal()
        return p2p_dist, None

    def _predict_sgn(
            self,
            source,
            target,
            cross_source,
            cross_target,
    ):
        source_subem = copy.deepcopy(source)
        target_subem = copy.deepcopy(target)
        for i in range(self.k):
            adjacency_source = adjacency_matrix(source_subem, mode="gaussian", method="knn")
            _, dd_source = csgraph_laplacian(
                adjacency_source, normed=False, return_diag=True
            )
            adjacency_target = adjacency_matrix(target_subem, mode="gaussian", method="knn")
            _, dd_target = csgraph_laplacian(
                adjacency_target, normed=False, return_diag=True
            )
            # Stochastically select target points for cross connections
            n_cross = int(self.cross * len(target))
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
            # print("Coupling graphs ...")
            # print(":: Number of cross connections", n_cross)
            # Compute coupled adjacency
            adjacency_coupled = couple_adjacency_matrices(
                adjacency_a=adjacency_source + np.identity(adjacency_source.shape[0]),
                adjacency_b=adjacency_target + np.identity(adjacency_target.shape[0]),
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
        mean_dist = np.mean(p2p_dist)
        # centroid_source = np.median(source_subem, axis=0)
        # centroid_target = np.median(target_subem, axis=0)
        # mean_dist = cosine_distances(np.asarray(centroid_source).reshape(1, -1), np.asarray(centroid_target).reshape(1, -1))
        # mean_dist = mean_dist[0, 0]
        return p2p_dist, mean_dist

    def _predict_cgn(
            self,
            source,
            target,
            cross_source,
            cross_target,
    ):
        source_subem = copy.deepcopy(source)
        target_subem = copy.deepcopy(target)
        for i in range(self.k):
            adjacency_source = adjacency_matrix(source_subem, mode="gaussian", method="knn")
            _, dd_source = csgraph_laplacian(
                adjacency_source, normed=False, return_diag=True
            )
            adjacency_target = adjacency_matrix(target_subem, mode="gaussian", method="knn")
            _, dd_target = csgraph_laplacian(
                adjacency_target, normed=False, return_diag=True
            )
            # Stochastically select target points for cross connections
            n_cross = int(self.cross * len(target))
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
            # print("Coupling graphs ...")
            # print(":: Number of cross connections", n_cross)
            # Compute coupled adjacency
            adjacency_coupled = couple_adjacency_matrices(
                adjacency_a=adjacency_source + np.identity(adjacency_source.shape[0]),
                adjacency_b=adjacency_target + np.identity(adjacency_target.shape[0]),
                inds_a=cross_source.reshape(-1),
                inds_b=cross_target.reshape(-1),
                dist=sp.diags(dist.reshape(-1)),
                return_uncoupled=False
            )
            coupled_laplacian = csgraph_laplacian(adjacency_coupled, normed=True)
            coupled_s = np.identity((coupled_laplacian.shape)[0]) - coupled_laplacian
            coupled_maps = np.vstack((source, target))
            coupled_embeddings = coupled_s * coupled_maps
            coupled_embeddings = ReluActivation(coupled_embeddings)
            # Split them
            source_embeddings = coupled_embeddings[:len(source)]
            target_embeddings = coupled_embeddings[len(source):]
            # Embedding restriceted to cross nodes
            source_subemb = source_embeddings[cross_source.reshape(-1), :]
            target_subemb = target_embeddings[cross_target.reshape(-1), :]
        # point to point distance
        p2p_dist = cosine_distances(np.asarray(source_subemb), np.asarray(target_subemb)).diagonal()
        mean_dist = np.mean(p2p_dist)
        # centroid_source = np.median(source_subem, axis=0)
        # centroid_target = np.median(target_subem, axis=0)
        # mean_dist = cosine_distances(np.asarray(centroid_source).reshape(1, -1), np.asarray(centroid_target).reshape(1, -1))
        # mean_dist = mean_dist[0, 0]
        return p2p_dist, mean_dist
