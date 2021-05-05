# Import packages
import numpy as np
from spektral.data import DisjointLoader
from spektral.datasets import QM9


def load_qm9(amount=1000):
    """
    Load the QM9 dataset. More information on the dataset, i.e. node features, edge features etc is available at
    https://graphneural.network/datasets/

    :param amount: number of graphs (molecules) to load. If None the whole dataset is loaded. default_value=1000
    :return: the dataset along with the node, edge features and the target dimension.
    """
    # Set amount=None to load the whole dataset
    dataset = QM9(amount=amount)
    # Dimension of node features
    F = dataset.n_node_features
    # Dimension of edge features
    S = dataset.n_edge_features
    # Dimension of the target
    n_out = dataset.n_labels
    return dataset, F, S, n_out


def split_dataset(dataset, batch_size, epochs, train_percentage=0.9):
    # Shuffle the indexes
    indexes = np.random.permutation(len(dataset))
    # Select the amount of samples to keep for training
    split = int(train_percentage * len(dataset))
    # Cut the shuffled indexes array after "split" samples
    idx_tr, idx_te = np.split(indexes, [split])
    # Divide the datasets
    dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]
    # Create the iterators needed during the training and testing phases
    loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
    loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)
    return loader_tr, loader_te
