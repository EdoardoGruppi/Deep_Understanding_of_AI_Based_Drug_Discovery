# Import packages
from itertools import chain
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import numpy as np
import os
from pandas import concat, read_csv
from Modules.metrics import sa_score, qed_score, get_activity
from Modules.config import *
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from Modules.metrics import validity
from Models.cmp_model import ChEMBL27


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
    and vice versa are created and outputted as well. Function used by CharRNN for preparing data.

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


def gen_data_vae(data, char_int_dict, max_length):
    """
    Create a one-hot representation of the whole dataset where the rows, columns and depth correspond to the smiles
    strings, their chars location and the chars types respectively.

    :param data: data to represent as a one-hot encoded 3D matrix.
    :param char_int_dict: dictionary to convert chars into integers.
    :param max_length: max sequence length in data.
    :return: the examples represented by one-hot encodings vectors.
    """
    one_hot = np.zeros((data.shape[0], max_length + 1, len(char_int_dict)), dtype=np.int8)
    # For each smiles string encode its chars
    for row, smile in enumerate(data):
        for col, char in enumerate(smile):
            one_hot[row, col, char_int_dict[char]] = 1
        # Add the stop char encoding
        one_hot[row, len(smile):, char_int_dict['E']] = 1
    # Return two one-hot encoding 3D matrices as examples and labels
    return one_hot


def load_datasets_smiles_vae(filename, samples, test_size=0.1):
    """
    Retrieves the data from the filename.txt file and, after converting them into a new representation, the function
    returns the samples of the training and testing datasets. A dictionary to map chars to integers is created and
    outputted as well. Function used by SmilesVAE for preparing data.

    :param filename: name of the txt file from which to load data.
    :param samples: number of samples to consider. If None all the samples are used. default_value=None
    :param test_size: percentage size of samples dedicated to the testing phase. default_value=0.2
    :return: the training and testing samples along with the aforementioned dictionary.
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
    # Max sequence length in the data loaded
    length = max([len(seq) for seq in train_data])
    # Convert data representations in one-hot encoding
    data = gen_data_vae(train_data, char_to_int, length)
    # Shuffle and divide the dataset
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42, shuffle=True)
    return train_data, test_data, char_to_int, length + 1


def create_cond_dataset(filename, new_filename, activity=False):
    """
    Given a dataset of smiles string, this function adds property scores with \t in between. If required,
    the activity of each molecule is reported as well. 1 means the molecule is active towards the GSK-3beta protein.

    :param filename: only the name of the file where the original dataset is saved.
    :param new_filename: name of the file in which the new dataset is saved.
    :param activity: boolean, if True the activity is also inserted.
    :return:
    """
    # Get the paths of the old and new dataset filenames
    path = os.path.join(base_dir, filename)
    new_path = os.path.join(base_dir, new_filename)
    # Transform every line of the file into an item of a list
    with open(path) as f:
        data = f.read().splitlines()
    # Create the new_file
    new_file = open(new_path, 'w')
    # If activity is required
    if activity:
        # Target found from the website:
        # https://www.ebi.ac.uk/chembl/g/#search_results/all/query=Glycogen%20synthase%20kinase%203%20beta
        # Find which molecules are considered active against the target selected
        activities = get_activity(path, confidence=90, targets=['CHEMBL262'], activity=['active'])
        # Get the list of active molecules
        activities = activities['Smiles'].to_list()
        # For every string in the old dataset
        for item in tqdm(data):
            # Retrieve the corresponding molecule object
            mol = MolFromSmiles(item)
            # Compute the SA and QED scores
            sa = sa_score(mol)
            qed = qed_score(mol)
            # Compute the activity value (from boolean to integer)
            value = 1 * (item in activities)
            # Save the calculation within the new file
            new_file.write(f'{item}\t{qed}\t{sa}\t{value}\n')
    else:
        # For every string in the old dataset
        for item in tqdm(data):
            # Retrieve the corresponding molecule object
            mol = MolFromSmiles(item)
            # Compute the SA and QED scores
            sa = sa_score(mol)
            qed = qed_score(mol)
            # Save the calculation within the new file
            new_file.write(f'{item}\t{qed}\t{sa}\n')
    # Close the new file to save all the changes
    new_file.close()


def dataset_activity(dataset_file, targets=None):
    """
    Get the activity of all the molecules in the test_file towards the selected targets.

    :param dataset_file: path to the file containing smiles strings to evaluate, one per line.
    :param targets: if None the prediction is computed for all the 500 available targets. Otherwise,
        it corresponds to a list of target_ids on which to evaluate the given molecule.
    :return: a dataframe with all the results.
    """
    # Retain only the set of unique valid generated molecules
    _, gen_molecules = validity(dataset_file)
    gen_molecules = set(gen_molecules)
    # Instantiate ChEMBL27 model
    model = ChEMBL27()
    # Get a list of dataframes as a result
    results = []
    for mol in tqdm(gen_molecules):
        # Get the results and keep only some information
        prediction = model.predict_activity(mol, targets)
        # Keep track of the molecule
        prediction['Smiles'] = mol
        results.append(prediction)
    # Concatenate all the dataframes
    results = concat(results, ignore_index=True)
    results.to_csv(os.path.join(base_dir, 'chembl_CFD_activity.csv'))
    return results


def activity_subset(filename, activity=('active', 'both'), confidence=70):
    dataframe = read_csv(os.path.join(base_dir, filename))
    print(f'Original dataframe length: {len(dataframe)}')
    dataframe = dataframe[dataframe[f'{confidence}%'].isin(activity)]
    print(f'New dataset length: {len(dataframe)}')
    with open(os.path.join(base_dir, 'active_chembl_27.txt'), 'w') as f:
        smiles = list(dataframe['Smiles'])
        for smile in smiles:
            f.write(smile + '\n')
