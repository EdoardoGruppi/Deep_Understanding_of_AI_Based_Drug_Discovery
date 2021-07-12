# Import packages
from rdkit.Chem import Crippen, Descriptors, QED, MolFromSmiles, AddHs, MolToSmiles, MolFromSmarts
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Contrib.NP_Score import npscorer
from rdkit import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.stats import entropy, gaussian_kde
import seaborn as sns
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
import matplotlib.pyplot as plt
import fcd
import numpy as np
import os
import random
import pkgutil
import tempfile
from tqdm import tqdm
from Modules.config import *
from pandas import DataFrame, Series, concat
from Models.cmp_model import ChEMBL27


def validity(test_file, output=False):
    """
    Computes the ratio of valid generated molecules.

    :param test_file: file path containing smiles strings of the generated molecules, one for line.
    :param output: boolean. If True the computed values are also printed on the console. default_value=False
    :return: the ratio of valid molecules as well as the list of valid smiles molecules.
    """
    # Read the .txt file returning a list of all the generated molecules
    with open(test_file) as f:
        test_list = f.read().splitlines()
    # List of valid molecules
    valid_molecules = [smiles for smiles in test_list if is_valid(smiles)]
    # Compute the ratio
    ratio = len(valid_molecules) / len(test_list)
    if output:
        # Console output
        print(f'The {ratio * 100:.2f} % of molecules are valid.\n')
    return ratio, valid_molecules


def is_valid(smiles):
    """
    Check if a smiles string corresponds to a valid molecule.

    :param smiles: the smiles string to evaluate.
    :return: boolean describing if the molecule is valid.
    """
    # convert the smiles string into a mol object
    mol = MolFromSmiles(smiles)
    # Return True if the condition is met
    return smiles != '' and mol is not None and mol.GetNumAtoms() > 0


def novelty(train_file, test_file):
    """
    Computes the novelty of the generated molecules. It corresponds to the percentage of unique created molecules
    that are not seen during training. The files should be two .txt files comprising the molecules used
    during training and obtained throughout the testing phase. Novelty formula taken from the MOSES benchmark.

    :param train_file: file path containing smiles strings of the training molecules, one for line.
    :param test_file: file path containing smiles strings of the generated molecules, one for line.
    :return: the novelty score.
    """
    # Get the list of valid molecules in the training dataset
    _, train_set = validity(train_file)
    # Consider each molecule only once
    train_set = set(train_set)
    # Get the list of valid molecules in the training dataset
    _, test_set = validity(test_file)
    # Consider each molecule only once
    test_set = set(test_set)
    # Compute the novelty ratio
    novelty_ratio = len(test_set - train_set) / len(test_set)
    # Console output
    print(f'Novelty is computed on the unique valid smiles strings passed for the training and test datasets.',
          f'\nNovelty ratio: {novelty_ratio * 100:.2f} %\n')
    return novelty_ratio


def uniqueness(test_file):
    """
    Computes the uniqueness of the generated molecules. The uniqueness ratio is computed only on the valid molecules.

    :param test_file: file path containing smiles strings of the generated molecules, one for line.
    :return: the uniqueness score computed discarding the invalid molecules.
    """
    # Check the validity of the smiles strings
    ratio, test_list = validity(test_file)
    # Retain only unique generated molecules
    test_set = set(test_list)
    # Compute the uniqueness
    uniqueness_ratio = len(test_set) / len(test_list)
    # Console output
    print(f'Uniqueness is computed on the valid {ratio * 100:.2f} % of the smiles strings passed.',
          f'\nUniqueness ratio: {uniqueness_ratio * 100:.2f} %\n')
    return uniqueness_ratio


def log_p(mol):
    """
    Computes Wildman-Crippen LogP value. More info at https://bit.ly/3h589Ct.

    :param mol: input rdkit molecule object.
    :return: the LogP value of the molecule.
    """
    return Crippen.MolLogP(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.

    :param mol: input rdkit molecule object.
    :return: the molecule weight.
    """
    return Descriptors.MolWt(mol)


def qed_score(mol):
    """
    Computes the Quantitative Estimation of Drug-likeness score. The empirical rationale of the QED measure reflects
    the underlying distribution of molecular properties including molecular weight, logP, topological polar surface
    area, number of hydrogen bond donors and acceptors, the number of aromatic rings and rotatable bonds,
    and the presence of unwanted chemical functionalities.

    :param mol: input rdkit molecule object.
    :return: the qed score of the molecule.
    """
    return QED.qed(mol)


def sa_score(mol):
    """
    Computes the Synthetic Accessibility score.

    :param mol: input rdkit molecule object.
    :return: the SA score of the molecule.
    """
    return sascorer.calculateScore(mol)


# Compute only once the second argument required to compute the NP score.
np_f_score = npscorer.readNPModel()


def np_score(mol):
    """
    Computes the Natural Product-likeness.

    :param mol: input rdkit molecule object.
    :return: the NP score of the molecule.
    """
    return npscorer.scoreMol(mol, fscore=np_f_score)


def get_num_atoms(mol):
    """
    Computes the number of heavy atoms contained in the mol object.

    :param mol: input rdkit molecule object.
    :return: the NP score of the molecule.
    """
    return mol.GetNumAtoms()


def property_distributions(list_files, list_names, prop='log_p', txt=True):
    """
    Compares the distribution of the property values obtained with different datasets.

    :param list_files: list of txt paths where the smiles string must be read. If the txt variable is False then
        list_files already corresponds to a list of lists of smiles strings.
    :param list_names: list of names to associate with each file inside the plots.
    :param prop: string defining which property to evaluate. It can be [logP, MW, QED, SA, NP]. default_value='log_p'
    :param txt: boolean. If True the list_files variable is a list of txt file names. Otherwise, the list_files
        variable corresponds to a list of lists of smiles strings. default_value=True
    :return: the comparison between the distributions plots.
    """
    # Dictionary of functions that can be called to compute the molecule properties
    dictionary = {'logP': log_p, 'MW': weight, 'QED': qed_score, 'SA': sa_score, 'NP': np_score, 'Atoms': get_num_atoms}
    # Select the chosen property to evaluate
    prop_function = dictionary[prop]
    # If True then list_files is a list of txt paths where the smiles string must be read.
    if txt:
        # The list of txt files are converted to a list of lists of smiles strings.
        list_files = [validity(file_txt)[1] for file_txt in list_files]
    # All the smiles string are transformed in mol objects
    list_files = [list(map(MolFromSmiles, molecules_list)) for molecules_list in list_files]
    # Compute the property for each molecule
    scores = [list(map(prop_function, molecules_list)) for molecules_list in list_files]
    # Plot the distributions
    sns.set()
    # Plot a distribution for each given txt file
    for score in scores:
        sns.kdeplot(data=score, legend=False)
    # Insert the property name as x_label
    plt.xlabel(prop, fontsize=13)
    # Insert a legend to associate each file with its distribution plot
    plt.legend(loc='best', labels=list_names)
    plt.tight_layout()
    plt.show()


def get_random_subset(dataset, subset_size):
    """
    Create a random subset of some dataset.

    :param dataset: original training dataset.
    :param subset_size: target size of the subset, described as number of samples.
    :return: return a random subset of a given dataset.
    """
    if len(dataset) < subset_size:
        raise Exception(f'The dataset to extract a subset from is too small: {len(dataset)} < {subset_size}')
    subset = random.sample(dataset, subset_size)
    return subset


def frechet_distance(train_file, test_file, chem_net_model_filename='ChemNet_v0.13_pretrained.h5', sample_size=10000):
    """
    Computes the Frechet distance between the training and test data distributions. With very large datasets it could
    be necessary to limit the measurements to smaller ensemble fo molecules.

    :param train_file: file path containing smiles strings of the training molecules, one for line.
    :param test_file: file path containing smiles strings of the generated molecules, one for line.
    :param chem_net_model_filename: name of the file for trained ChemNet model. Must be present in the 'fcd' package,
        since it will be loaded directly from there.
    :param sample_size: size of the subset sampled from the reference set. default_value=10000
    :return: the FCD scores as presented in the original paper and in GuacaMol. While in the first case the lower
        values are better, in the second case they are as much better as they are close to 1.
    """
    # Load the ChemNet model
    chem_net = load_chem_net(chem_net_model_filename)
    # Retrieve only the valid molecules of the given .txt files
    _, reference_molecules = validity(train_file)
    _, generated_molecules = validity(test_file)
    # Get a subset of the reference dataset
    reference_molecules = get_random_subset(reference_molecules, sample_size)
    # Calculate the distribution statistics for each dataset
    mu_ref, cov_ref = calculate_distribution_statistics(chem_net, reference_molecules)
    mu, cov = calculate_distribution_statistics(chem_net, generated_molecules)
    # Compute the FCD distance given the statistics of the two distributions
    FCD_score = fcd.calculate_frechet_distance(mu1=mu_ref, mu2=mu, sigma1=cov_ref, sigma2=cov)
    # In GuacaMol the FCD score is transformed so that it goes from zero to one
    fcd_score = np.exp(-0.2 * FCD_score)
    print(f'Original Frechet distance: {FCD_score:.8f}',
          f'\nFCD GuacaMol score: {fcd_score:.14f}\n')
    return FCD_score, fcd_score


def calculate_distribution_statistics(model, molecules):
    """
    Computes the statistic of a dataset distribution.

    :param model: a ChemNet loaded model.
    :param molecules: the list of molecules to evaluate with the model to then get the statistics.
    :return:
    """
    # Get the predictions of the model made on the given molecules
    gen_mol_act = fcd.get_predictions(model, molecules)
    # Compute the mean of the distribution
    mu = np.mean(gen_mol_act, axis=0)
    # Compute the covariance matrix
    cov = np.cov(gen_mol_act.T)
    return mu, cov


def load_chem_net(chem_net_model_filename):
    """
    Loads the ChemNet model from the file specified in the init function. This file lives inside a package but to use
    it, it must always be an actual file. Re-implementation inspired by the GuacaMol code.
    The safest way to proceed is therefore:
    1. read the file with pkgutil
    2. save it to a temporary file
    3. load the model from the temporary file

    :param chem_net_model_filename: name of the file for trained ChemNet model. Must be present in the 'fcd' package,
        since it will be loaded directly from there.
    :return: the loaded model.
    """
    # Get a resource from a package.
    model_bytes = pkgutil.get_data(package='fcd', resource=chem_net_model_filename)
    # Return the name of the directory used for temporary files.
    tmpdir = tempfile.gettempdir()
    # Create the path where to save the model
    model_path = os.path.join(tmpdir, chem_net_model_filename)
    # Copy the binary data in the new binary file
    with open(model_path, 'wb') as f:
        f.write(model_bytes)
    # Returns the loaded ChemNet model.
    return fcd.load_ref_model(model_path)


def filter_molecules(file, allowed=None, filters='pains_filters.txt', check='filters', qed=None, sa=None):
    """
    Checks if the molecules generated pass the MCF and/or PAINS filters, have only allowed atoms and are not charged.

    :param file: file path containing smiles strings of the molecules to evaluate, one for line.
    :param allowed: list of allowed atoms. By default, they are {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}.
    :param filters: names of the file containing smarts strings of the filters to use. 'pains_filters.txt' for pains
        filters, 'mcf_filters.txt' for mfc filters, 'all_filters.txt' for both.
    :param check: variable defining which type of check to perform. It can be partial ('filters'), i.e. only filters,
        or 'complete'.
    :param qed: minimum qed value required to pass the check. If None, it is not considered. default_value=None
    :param sa: maximum sa value required to pass the check. If None, it is not considered. default_value=None
    :return: the list of smiles strings of the molecules that passed the check.
    """
    print(f'Filtering the generated molecules with {check} check. Filters selected are: {filters.split(".")[0]}.')
    # Select the required check function
    checks = {'complete': mol_passes_complete_check, 'filters': mol_passes_filters}
    check = checks[check]
    # Read the smarts strings of the filters and convert them to mol objects
    with open(os.path.join(base_dir, filters)) as f:
        filters = f.read().splitlines()
        filters = [MolFromSmarts(smart) for smart in filters]
    # List of allowed atoms
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H', 'I', 'P'}
    # Consider only the valid smiles strings within the list of molecules to test
    _, smiles_molecules = validity(file)
    # Convert them to mol objects
    molecules = [MolFromSmiles(smiles) for smiles in smiles_molecules]
    # Retain only the molecules that pass the check process
    molecules = [mol for mol in molecules if check(mol, filters, allowed, qed, sa)]
    # Represent them with smiles strings
    passed_molecules = [MolToSmiles(mol) for mol in molecules]
    print(f'{len(passed_molecules) * 100 / len(smiles_molecules):.2f}% of molecules successfully passed the check.\n')
    return passed_molecules


def mol_passes_complete_check(mol, filters, allowed=None, qed=None, sa=None):
    """
    Check if a molecule successfully pass a series of pre-defined conditions.

    :param mol: mol object to test.
    :param filters: list of filters (as mol objects) to consider in the evaluation.
    :param allowed: list of allowed atoms.
    :param qed: minimum qed value required to pass the check. If None, it is not considered. default_value=None
    :param sa: maximum sa value required to pass the check. If None, it is not considered. default_value=None
    :return: a boolean. True if the molecule passed successfully all the conditions.
    """
    # Return False if the mol object is None
    if mol is None:
        return False
    # Retrieve information about a molecule’s rings
    ring_info = mol.GetRingInfo()
    # Remove molecules containing cycles larger than 8 atoms
    if ring_info.NumRings() != 0 and any(len(x) >= 8 for x in ring_info.AtomRings()):
        return False
    # Adds hydrogen atoms to the graph of a molecule
    h_mol = AddHs(mol)
    # Remove molecules with charged atoms
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    # Remove molecules containing not allowed atoms
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    # Filter molecules by means of the given filters
    if any(h_mol.HasSubstructMatch(smarts) for smarts in filters):
        return False
    # Remove molecules whose conversion to smiles string is equal to None or ''
    smiles = MolToSmiles(mol)
    if smiles is None or len(smiles) == 0:
        return False
    # Remove molecules whose conversion to mol object returns None
    if MolFromSmiles(smiles) is None:
        return False
    # Rule out molecules whose QED value is less than the one required
    if qed is not None:
        if qed_score(mol) < qed:
            return False
    # Exclude molecules whose SA value is greater than the one required
    if sa is not None:
        if sa_score(mol) > sa:
            return False
    return True


def mol_passes_filters(mol, filters, qed=None, sa=None, allowed=None):
    """
    Check if a molecule successfully pass the filtering process.

    :param mol: mol object to test.
    :param filters: list of filters as mol objects to consider in the evaluation.
    :param qed: minimum qed value required to pass the check. If None, it is not considered. default_value=None
    :param sa: maximum sa value required to pass the check. If None, it is not considered. default_value=None
    :param allowed: not required.
    :return: a boolean. True if the molecule passed successfully all the conditions.
    """
    # Return False if the mol object is None
    if mol is None:
        return False
    # Adds hydrogen atoms to the graph of a molecule
    h_mol = AddHs(mol)
    # Filter molecules by means of the given filters
    if any(h_mol.HasSubstructMatch(smarts) for smarts in filters):
        return False
    if qed is not None:
        if qed_score(mol) < qed:
            return False
    if sa is not None:
        if sa_score(mol) > sa:
            return False
    return True


def internal_diversity(file, p=1):
    """
    Computes internal diversity as:  1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))

    :param file: path to the file containing smiles strings of the molecules to evaluate, one for line.
    :param p: power for averaging (mean x^p)^(1/p). default_value=1
    :return: the internal diversity score [0,1]. Higher values means higher diversity. in the generated set.
    """
    # Retain only the valid molecules
    _, molecules = validity(file)
    # Convert the molecules into Morgan fingerprints
    fingerprints = get_fingerprints(molecules)
    # Compute the internal diversity score
    internal_div = 1 - (average_agg_tanimoto(fingerprints, fingerprints, agg='mean', p=p))
    print(f'Internal diversity is: {internal_div:.4f}\n')
    return internal_div


def get_fingerprint(smiles, morgan_r=2, morgan_n=1024):
    """
    Transforms a smiles string into a Morgan fingerprint bit vector.

    :param smiles: smiles string to convert into a fingerprint.
    :param morgan_r: Morgan fingerprint radius. default_value=2
    :param morgan_n: the dimension of the Morgan fingerprint vector. default_value=1024
    :return: the related Morgan fingerprint.
    """
    # Convert the smiles string into a molecule
    molecule = MolFromSmiles(smiles)
    # Return a Morgan fingerprint for a molecule as a bit vector
    return np.asarray(Morgan(molecule, radius=morgan_r, nBits=morgan_n), dtype='uint8')


def get_fingerprints(smiles_mol_array):
    """
    Computes fingerprints of smiles.

    :param smiles_mol_array: smiles to convert into fingerprints.
    :return: the fingerprints of the smiles strings.
    """
    # Convert the list of smiles into an array
    smiles_mol_array = np.asarray(smiles_mol_array)
    # Keep note of the indexes of the duplicates and remove them
    smiles_mol_array, inv_index = np.unique(smiles_mol_array, return_inverse=True)
    # Compute all the fingerprints
    fingerprints = [get_fingerprint(mol) for mol in smiles_mol_array]
    # Stack them into a nd array
    fingerprints = np.vstack(fingerprints)
    # Return the fingerprints of all the smiles strings including of the duplicates
    return fingerprints[inv_index]


def average_agg_tanimoto(stock_vectors, gen_vectors, batch_size=100, agg='max', p=1):
    """
    For each molecule in gen_vectors finds closest molecule in stock_vectors. Returns average tanimoto score between
    these molecules.

    :param stock_vectors: numpy array of fingerprints (n_vectors x dim) of the first dataset.
    :param gen_vectors: numpy array of fingerprints (n_vectors x dim) of the second dataset.
    :param batch_size: batch_size of samples to simultaneously work with. default_value=100
    :param agg: type: of aggregator. It can be 'max' or 'mean'. default_value='max'
    :param p: power for averaging (mean x^p)^(1/p). default_value=1
    :return:
    """
    # Number of vectors in the two datasets
    num_gen_vectors = gen_vectors.shape[0]
    num_stock_vectors = stock_vectors.shape[0]
    # To be sure the agg parameter is correctly fulfilled
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    # Instantiate the arrays wherein the agg_tanimoto and total results will be saved
    agg_tanimoto = np.zeros(num_gen_vectors)
    total = np.zeros(num_gen_vectors)
    # Every batch_size vectors of the first dataset
    for j in range(0, num_stock_vectors, batch_size):
        x_stock = stock_vectors[j:j + batch_size]
        # Consider one batch of vectors of the second dataset after the other
        for i in range(0, num_gen_vectors, batch_size):
            y_gen = gen_vectors[i:i + batch_size]
            # Transpose the matrix related to the batch of the second dataset
            y_gen = y_gen.transpose()
            # Performs a matrix multiplication of the two matrices
            tp = np.matmul(x_stock, y_gen)
            # Compute the Tanimoto matrix
            jac = (tp / (x_stock.sum(axis=1, keepdims=True) + y_gen.sum(axis=0, keepdims=True) - tp))
            # Replace all the nan values, if any, of the matrix with ones
            jac[np.isnan(jac)] = 1
            # Compute the power of p of the Jaccard matrix
            if p != 1:
                jac = jac ** p
            if agg == 'max':
                # Update the agg_tanimoto variable. np.maximum compare two arrays and returns a new array containing
                # the element-wise maxima between the old values of the variable and the maxima column values of jac.
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(agg_tanimoto[i:i + y_gen.shape[1]], jac.max(axis=0))
            elif agg == 'mean':
                # Update the agg_tanimoto variable adding the column-wise sum of the jac values to the old values
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(axis=0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    # Perform the averaging process required by the Tanimoto formula
    if p != 1:
        agg_tanimoto = agg_tanimoto ** (1 / p)
    return np.mean(agg_tanimoto)


def snn_metric(file_ref, file_gen):
    """
    Computes the snn metric as: SNN = 1/|A| sum_{x in AxA} max{y in RxR} tanimoto(x, y). A metric DfD is also
    returned as: 1 - SNN.

    :param file_ref: path to the file containing the reference smiles strings to evaluate, one for line.
    :param file_gen: path to the file containing the generated smiles strings to evaluate, one for line.
    :return: the SNN and DfD scores [0,1]. If generated molecules are far from the reference molecules they SNN and DfD
        values are respectively closer to 0 and 1.
    """
    # Retain only the reference smiles strings that correspond to valid molecules
    _, ref_molecules = validity(file_ref)
    # Retain only the generated smiles strings that correspond to valid molecules
    _, gen_molecules = validity(file_gen)
    # Convert the molecules into Morgan fingerprints
    ref_fingerprints = get_fingerprints(ref_molecules)
    gen_fingerprints = get_fingerprints(gen_molecules)
    # Compute the internal diversity score
    snn = average_agg_tanimoto(ref_fingerprints, gen_fingerprints, agg='max', p=1)
    dfd = 1 - snn
    print(f'SNN metric is: {snn:.4f} - DfD is: {dfd:.4f}')
    return snn, dfd


def atom_occurrences(file, atoms_list=('C', 'O', 'N', 'Cl', 'F')):
    """
    Computes the atoms occurrences, namely how many times an atom is used in a dataset and in how many molecules it
    appears. The values are also normalised with the number of valid molecules within the dataset.

    :param file: path to the file containing smiles strings of the molecules to evaluate, one for line.
    :param atoms_list: list of atoms to evaluate. It can be computed using the get_atom_list function.
    :return: a summary of the atoms occurrences displayed on the console.
    """
    # Consider only the valid molecules
    _, molecules = validity(file)
    # Create an empty dataframe to count the occurrences of the atoms in each molecule
    dataframe = DataFrame(columns=atoms_list)
    # For every mol object...
    for molecule in tqdm(molecules):
        # Convert the SMILES string into a mol object
        mol = MolFromSmiles(molecule)
        # Get the list of atom symbols
        atoms = mol.GetAtoms()
        symbols = [atom.GetSymbol() for atom in atoms]
        # Append a new row in the dataframe with the number of times each atom is used
        dataframe = dataframe.append(Series([symbols.count(atom_type) for atom_type in atoms_list],
                                            index=dataframe.columns), ignore_index=True)
    # Get the number of times every atom is used in the dataset
    sum_list = DataFrame.sum(dataframe, axis=0)
    # Count in how many molecules each atom appears
    count_list = DataFrame.sum(dataframe.astype(bool), axis=0)
    # Display the summary with non-normalised values
    print('\nAtom occurrences ' + '-' * 40)
    for atom, sum_value, count in zip(atoms_list, sum_list, count_list):
        print(f'Atom: {atom:2} - Total occurrences: {sum_value:8} - In {count:7} molecules')
    # Compute and display the summary with the normalised values
    sum_list = sum_list.values / len(molecules)
    count_list = count_list.values / len(molecules)
    print('\nNormalised values')
    for atom, sum_value, count in zip(atoms_list, sum_list, count_list):
        print(f'Atom: {atom:2} - Total occurrences: {sum_value:8.4f} - In {count:7.4f} molecules')
    print('-' * 57 + '\n')


def get_atom_list(file):
    """
    Get the list of atoms that are present in a particular dataset.

    :param file: path to the file containing smiles strings of the molecules to evaluate, one for line.
    :return: the list of atoms found in the dataset.
    """
    # Consider only the valid molecules
    _, molecules = validity(file)
    # Convert each string into a mol object
    molecules = [MolFromSmiles(molecule) for molecule in molecules]
    # Instantiate an empty list
    atoms_list = []
    # For each molecule in the dataset
    for mol in molecules:
        # Get the atoms symbols
        atoms = mol.GetAtoms()
        # And compute an union operation with the list of atoms already discovered
        atoms_list = set().union(atoms_list, [atom.GetSymbol() for atom in atoms])
    return atoms_list


def kl_divergence(train_file, test_file, subset_size=None):
    """
    Compute the KL divergence between two data distributions, i.e. their descriptor values and their internal
    pairwise similarity.  With very large datasets it could be necessary to limit the measurements to smaller
    ensemble fo molecules.

    :param train_file: path to the file containing smiles strings of the reference molecules, one for line.
    :param test_file: path to the file containing smiles strings of the generated molecules to evaluate, one for line.
    :param subset_size: size of the sampling performed on the training dataset. default_value=None
    :return: the KL divergence score.
    """
    # Retain only the set of unique valid reference molecules
    _, ref_molecules = validity(train_file)
    ref_molecules = set(ref_molecules)
    # Retain only the set of unique valid generated molecules
    _, gen_molecules = validity(test_file)
    gen_molecules = set(gen_molecules)
    # If it is None set subset_size to the length of the generated set
    if subset_size is None:
        subset_size = len(gen_molecules)
    # Get only a subset of the training dataset
    ref_molecules = get_random_subset(ref_molecules, subset_size)
    # List of descriptors to which evaluate
    descriptors = ['BertzCT', 'MolLogP', 'MolWt', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds',
                   'NumAliphaticRings', 'NumAromaticRings']
    # Calculate the descriptors, which are np.arrays of size n_samples x n_descriptors
    ref_descriptors = calculate_descriptors(ref_molecules, descriptors)
    gen_descriptors = calculate_descriptors(gen_molecules, descriptors)
    # Object where the kl values will be saved
    kl_divs = {}
    # Calculate the kl divergence for the float valued descriptors
    for i in range(4):
        # Compute the kl divergence value for the distributions related to each descriptor
        kl_div = continuous_kl_div(x_baseline=ref_descriptors[:, i], x_sampled=gen_descriptors[:, i])
        # Save the result obtained
        kl_divs[descriptors[i]] = kl_div
    # Calculate the kl divergence for the int valued descriptors
    for i in range(4, 9):
        # Compute the kl divergence value for the distributions related to each descriptor
        kl_div = discrete_kl_div(x_baseline=ref_descriptors[:, i], x_sampled=gen_descriptors[:, i])
        # Save the result obtained
        kl_divs[descriptors[i]] = kl_div
    # Compute the internal pairwise similarity matrix for the reference molecules
    ref_similarity = calculate_internal_pairwise_similarities(ref_molecules)
    # Compute the maximum values within each row of the matrix
    ref_similarity = np.max(ref_similarity, axis=1)
    # Compute the internal pairwise similarity matrix for the generated molecules
    gen_similarity = calculate_internal_pairwise_similarities(gen_molecules)
    # Compute the maximum values within each row of the matrix
    gen_similarity = np.max(gen_similarity, axis=1)
    # Calculate the kl divergence value between the two similarity distributions
    kl_divs['internal_similarity'] = continuous_kl_div(x_baseline=ref_similarity, x_sampled=gen_similarity)
    # Each KL divergence value is transformed to be in [0, 1].
    partial_scores = [np.exp(-score) for score in kl_divs.values()]
    # Then their average delivers the final score.
    score = sum(partial_scores) / len(partial_scores)
    print(f'KL divergence score is: {score:10.4f}\n')
    return score


def calculate_descriptors(smiles, descriptors):
    """
    Compute the descriptors information for a list of molecules.

    :param smiles: list of smiles strings of the molecules to evaluate.
    :param descriptors:  list of strings defining the descriptor to use.
    :return:
    """
    # Initialise empty list where to append the results
    output = []
    for smiles_string in smiles:
        # For each molecule compute the descriptors information
        molecule_info = _calculate_descriptors(smiles_string, descriptors)
        # Append the result only if it is not None
        if molecule_info is not None:
            output.append(molecule_info)
    # Return the list converted into array
    return np.array(output)


def _calculate_descriptors(smiles, descriptors):
    """
    Compute the descriptor for a provided molecule. The array values returned is cleaned from all the non finite
    values if present.

    :param smiles: a smiles string defining the smiles string to evaluate.
    :param descriptors: list of strings defining the descriptor to use.
    :return: an array with all the descriptors information for the provided molecule.
    """
    # Object to calculate descriptors for the provided molecule
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
    # Convert the miles string into a mol object
    mol = MolFromSmiles(smiles)
    # Calculate all descriptors for a given molecule
    mol_info = calc.CalcDescriptors(mol)
    # Convert the result into an array
    mol_info = np.array(mol_info)
    # Keep track of the elements that are not finite (inf, nan)
    mask = np.isfinite(mol_info)
    # If one or more elements are not finite print a warning
    if (mask == 0).sum() > 0:
        print(f'{smiles} contains an NAN physchem descriptor')
        # Then replace all the non finite values with zeros
        mol_info[~mask] = 0
    return mol_info


def continuous_kl_div(x_baseline, x_sampled):
    """
    Computes the continuous Kullback–Leibler divergence (also called relative entropy) as a measure of how one
    probability  distribution is different from a second reference probability distribution.

    :param x_baseline: a numpy array of data on which to compute the first distribution (with pdf).
    :param x_sampled: a numpy array of data on which to compute the second distribution (with pdf).
    :return: the KL divergence value.
    """
    # Representation of kernel-density estimates using Gaussian kernels
    kde_P = gaussian_kde(dataset=x_baseline)
    kde_Q = gaussian_kde(dataset=x_sampled)
    # Return evenly spaced numbers over a specified interval
    x_eval = np.linspace(start=np.min(np.hstack([x_baseline, x_sampled])),
                         stop=np.max(np.hstack([x_baseline, x_sampled])), num=1000)
    # Evaluate the estimated pdf on a set of points
    P = kde_P(x_eval) + 1e-10
    Q = kde_Q(x_eval) + 1e-10
    # Return the relative entropy of a distribution for given probability values with respect to a given sequence
    return entropy(P, Q)


def discrete_kl_div(x_baseline, x_sampled):
    """
    Computes the discrete Kullback–Leibler divergence (also called relative entropy) as a measure of how one probability
    distribution is different from a second reference probability distribution.

    :param x_baseline: a numpy array of data on which to compute the first distribution (with histograms).
    :param x_sampled: a numpy array of data on which to compute the second distribution (with histograms).
    :return: the KL divergence value.
    """
    # Compute the histogram of a set of data. The result is the value of the probability density function at the bin,
    # normalized such that the integral over the range is 1.
    P, bins = np.histogram(x_baseline, bins=10, density=True)
    Q, _ = np.histogram(x_sampled, bins=bins, density=True)
    P += 1e-10
    Q += 1e-10
    # Return the relative entropy of a distribution for given probability values with respect to a given sequence
    return entropy(P, Q)


def calculate_internal_pairwise_similarities(smiles_list):
    """
    Computes the pairwise similarities of the provided list of smiles against itself.

    :param smiles_list: a collection of smiles strings.
    :return: symmetric matrix of pairwise similarities. Diagonal is set to zero.
    """
    # Print a notification if the dataset of smiles is particularly large
    if len(smiles_list) > 10000:
        print(f'Calculating internal similarity on large set of SMILES strings ({len(smiles_list)})')
    # Convert the smiles strings into mol objects
    molecules = [MolFromSmiles(smiles) for smiles in smiles_list]
    # And the mol objects into fingerprints
    fingerprints = [Morgan(mol, radius=2, nBits=4096) for mol in molecules]
    n_fingerprints = len(fingerprints)
    # Instantiate a square table where to save the similarity values
    similarities = np.zeros((n_fingerprints, n_fingerprints))
    # For each
    for i in range(1, n_fingerprints):
        # Compute the Tanimoto similarity between the current and the previous fingerprints
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        # Save the results in the symmetric matrix
        similarities[i, :i] = sims
        similarities[:i, i] = sims
    return similarities


def get_activity(test_file, confidence, targets=None, activity=('active', 'both')):
    """
    Get the activity of all the molecules in the test_file towards the selected targets.

    :param test_file: path to the file containing smiles strings of the generated molecules to evaluate, one for line.
    :param confidence: the level of confidence of the prediction. If the can be 70, 80 or 90.
    :param targets: if None the prediction is computed for all the 500 available targets. Otherwise,
        it corresponds to a list of target_ids on which to evaluate the given molecule.
    :param activity: keep track only this kind of activity. It can be ['active', 'both', 'inactive', 'empty'].
    :return: a dataframe with all the results.
    """
    # Retain only the set of unique valid generated molecules
    _, gen_molecules = validity(test_file)
    gen_molecules = set(gen_molecules)
    # Number of unique valid molecules
    n_unique_valid_molecules = len(gen_molecules)
    # Instantiate ChEMBL27 model
    model = ChEMBL27()
    # Get a list of dataframes as a result
    results = []
    for mol in tqdm(gen_molecules):
        # Get the results and keep only some information
        prediction = model.predict_activity(mol, targets)
        prediction = prediction[['Target_chembl_id', 'Organism', 'Pref_name', f'{confidence}%']]
        # Keep track of the molecule
        prediction['Smiles'] = mol
        # Delete all the dataframe rows where the molecule is not active
        prediction = prediction[prediction[f'{confidence}%'].isin(activity)]
        # Do not append the result if the dataframe is empty
        if prediction.shape[0] > 0:
            results.append(prediction)
    # Concatenate all the dataframes
    results = concat(results, ignore_index=True)
    print('Results of the targets activity predictions ' + '-' * 8)
    print(results.value_counts(['Target_chembl_id', f'{confidence}%']))
    print('\nNormalised values')
    print(results.value_counts(['Target_chembl_id', f'{confidence}%']) / n_unique_valid_molecules)
    print('-' * 57 + '\n')
    return results
