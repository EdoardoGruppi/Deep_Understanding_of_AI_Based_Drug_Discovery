# Import packages
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer


def log_p(mol):
    """
    Computes Wildman-Crippen LogP value. More info at https://bit.ly/3h589Ct.

    :param mol: input rdkit molecule object.
    :return: the LogP value of the molecule.
    """
    return Crippen.MolLogP(mol)


def sa_score(mol):
    """
    Computes the Synthetic Accessibility score.

    :param mol: input rdkit molecule object.
    :return: the SA score of the molecule.
    """
    return sascorer.calculateScore(mol)


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


def novelty(train_file, test_file):
    """
    Computes the novelty of the generated molecules. It corresponds to the percentage of unique created molecules
    that are not seen during training. The files should be two .txt files comprising the molecules used
    during training and obtained throughout the testing phase. Note: to get a fairer result it is probably preferable to
    exclude, if saved, the invalid molecules before.

    :param train_file: file path containing smiles strings of the training molecules, one for line.
    :param test_file: file path containing smiles strings of the generated molecules, one for line.
    :return: the novelty score.
    """
    with open(train_file) as f:
        train_set = set(f.read().splitlines())
    with open(test_file) as f:
        test_set = set(f.read().splitlines())
    return len(test_set - train_set) / len(test_set)


def uniqueness(test_file):
    """
    Computes the uniqueness of the generated molecules.

    :param test_file: file path containing smiles strings of the generated molecules, one for line.
    :return: the uniqueness score.
    """
    with open(test_file) as f:
        test_list = f.read().splitlines()
        test_set = set(test_list)
    return len(test_set) / len(test_list)


# filters
# activity
