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


def load_dataset(dataset, node_features, batch_size, epochs, edge_features):
    """
    Load the dataset as iterator objects. The function performs some operation before working on the passed dataset.

    :param dataset: a Spektral dataset object.
    :param node_features: number of node features to keep.
    :param batch_size: number of examples in every batch.
    :param epochs: number of epochs to run.
    :param edge_features: number of edge features to keep.
    :return: an iterator object.
    """
    # Remove all the unwanted features in the edge matrix
    if dataset.n_edge_features > edge_features:
        for item in dataset.graphs:
            item.e = item.e[:, :edge_features]
    # Remove all the unwanted features in the edge matrix
    if dataset.n_node_features > node_features:
        for item in dataset.graphs:
            item.x = item.x[:, :node_features]
    # Remove the y field of each graph object since not needed
    for item in dataset.graphs:
        item.y = None
    # Load the iterator object
    loader_tr, _ = split_dataset(dataset, batch_size, epochs, train_percentage=1)
    return loader_tr


def split_dataset(dataset, batch_size, epochs, train_percentage=0.9):
    """
    Splits the dataset when required.

    :param dataset: a Spektral dataset object.
    :param batch_size: number of examples in every batch.
    :param epochs: number of epochs to run.
    :param train_percentage: percentage defining the samples aimed to train the model.
    :return: two iterator objects.
    """
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
