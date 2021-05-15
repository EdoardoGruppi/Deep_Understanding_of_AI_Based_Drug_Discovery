# Import packages
from itertools import chain
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import numpy as np
import os
from Modules.config import *


def gen_data(data, char_int_dict, max_length):
    """
    Create a one-hot representation of the whole dataset where the rows, columns and depth correspond to the smiles
    strings, their chars location and the chars types respectively.

    :param data: data to represent as a one-hot encoded 3D matrix.
    :param char_int_dict: dictionary to convert chars into integers.
    :param max_length: max sequence length in data.
    :return: the examples and labels for training the CharRNN model.
    """
    one_hot = np.zeros((data.shape[0], max_length + 2, len(char_int_dict)), dtype=np.int8)
    # For each smiles string encode its chars
    for row, smile in enumerate(data):
        # Add the start char encoding
        one_hot[row, 0, char_int_dict['!']] = 1
        for col, char in enumerate(smile):
            one_hot[row, col + 1, char_int_dict[char]] = 1
        # Add the stop char encoding
        one_hot[row, len(smile) + 1:, char_int_dict['E']] = 1
    # Return two one-hot encoding 3D matrices as examples and labels
    return one_hot[:, 0:-1, :], one_hot[:, 1:, :]


def load_datasets(filename, samples=None, test_size=0.2):
    """
    Retrieves the data from the filename.txt file and, after converting them into a new representation, the function
    returns the samples and labels of the training and testing datasets. Two dictionaries to map chars to integers
    and vice versa are created and outputted as well.

    :param filename: name of the txt file from which to load data.
    :param samples: number of samples to consider. If None all the samples are used. default_value=None
    :param test_size: percentage size of samples dedicated to the testing phase. default_value=0.2
    :return: the training and testing samples and labels along with the dictionaries.
    """
    # Load data from the text file
    train_data = np.genfromtxt(os.path.join(base_dir, filename), dtype='U')
    # If required retain only the first n samples
    if not None:
        train_data = train_data[:samples]
    # Create a new list with all the unique chars in the data
    unique_chars = sorted(list(OrderedDict.fromkeys(chain.from_iterable(train_data))))
    # Add starting and stop letters to the list
    unique_chars.extend(['!', 'E'])
    # Map each char to int and vice versa through dictionaries
    char_to_int = dict((char, index) for index, char in enumerate(unique_chars))
    int_to_char = dict((index, char) for index, char in enumerate(unique_chars))
    # Max sequence length in the data loaded
    length = max([len(seq) for seq in train_data])
    # Convert data representations in one-hot encoding
    data, labels = gen_data(train_data, char_to_int, length)
    # Shuffle and divide the dataset
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size,
                                                                        random_state=42, shuffle=True)
    return train_data, test_data, train_labels, test_labels, char_to_int, int_to_char
