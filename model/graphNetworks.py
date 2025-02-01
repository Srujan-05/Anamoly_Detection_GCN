# from torch_geometric.nn import GCNConv
import numpy as np
import time
from utils.load_data import load_data
# from sklearn.naive_bayes import GaussianNB
from utils.io import load_tiff_pc
from utils.pre_processing import largest_connected_component
from manifold.utils import adjacency_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian


class SGN:
    def __init__(self, k):
        """
        :param k: k is the number of layers the SGN runs before running the classifier on the point-cloud
        """
        self.k = k
        self.classifier = np.zeros([1, 1])
        # self.classifier = GaussianNB()

    def train(self, training_sample, training_labels):
        start = time.time()
        for index in range(len(training_sample)):
            # print("Sample number {} file location {}".format(index+1, training_sample[index]))
            sample, sample_pca, sample_sampling = load_tiff_pc(training_sample[index])
            print("sample shape {}".format(sample.shape))
            # label = training_labels[index]
            adj_matix = adjacency_matrix(sample, mode="gaussian", method="knn")
            # sample = largest_connected_component(sample, adj=adj_matix, num_components=1)
            # adj_matix = adjacency_matrix(sample, mode="gaussian", method="knn")
            s_hat = csgraph_laplacian(adj_matix, normed=True) + np.identity(adj_matix.shape[0])
            h = np.power(s_hat, self.k) * sample
            print("h matrix shape {}".format(h.shape))
        print("The time taken: ", time.time() - start)


if __name__ == "__main__":
    training_sample, training_labels = load_data("../data/training")
    sgn = SGN(5)
    sgn.train(training_sample, training_labels)
