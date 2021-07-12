# Import packages
from rdkit.Chem import Draw, GetAdjacencyMatrix
from Modules.utils import *
import matplotlib.pyplot as plt
import pandas as pd
from Modules.metrics import *
import seaborn as sns


def show_mol(mol, name=False):
    """
    Displays a rdkit mol object.

    :param mol: rdkit mol object to visualize.
    :param name: if True search and writes the name of the molecule given.
    :return:
    """
    # Draw the molecule
    plt.imshow(Draw.MolToImage(mol))
    # Add the name if required
    if name:
        name = smiles_to_name(MolToSmiles(mol))
        plt.suptitle(name, fontsize=20)
    # Delete the axis and improve the visualization
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def molecule_info(mol, show=False):
    """
    Displays some general info about the given molecule.

    :param mol: molecule to visualize.
    :param show: boolean, if True an image of the molecule is generated.
    :return:
    """
    intro_string = f'{MolToSmiles(mol)} Molecule info ' + '.' * 50
    print('\n' + intro_string + '\n')
    print(mol.Debug())
    print('\nAdjacency Matrix: \n', GetAdjacencyMatrix(mol))
    print('.' * len(intro_string))
    if show:
        show_mol(mol)


def molecule_properties(mol):
    """
    Displays the main properties of a given molecule.

    :param mol: input molecule object.
    :return:
    """
    intro_string = f'{MolToSmiles(mol)} Molecule properties ' + '.' * 50
    print('\n' + intro_string,
          f'\n\nLog-p:    {log_p(mol):.4f}',
          f'\nWeight:   {weight(mol):.4f}',
          f'\nSA score: {sa_score(mol):.4f}',
          f'\nQED:      {qed_score(mol):.4f}',
          '\n' + '.' * len(intro_string))


def property_distributions_csv(list_files, list_names, prop='log_p'):
    """
    Compares the distribution of the property values obtained with different datasets.

    :param list_files: list of csv paths to read.
    :param list_names: list of names to associate with each file inside the plots.
    :param prop: string defining which property to evaluate. It can be [logP, MW, QED, SA, NP]. default_value='log_p'
    :return: the comparison between the distributions plots.
    """
    list_dataframes = []
    for file_path in list_files:
        df = pd.read_csv(file_path)
        list_dataframes.append(df)
    # Compute the property for each molecule
    scores = [list(df[prop]) for df in list_dataframes]
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


def property_histograms_csv(list_files, list_names, prop='log_p'):
    """
    Compares the distribution of the property values obtained with different datasets but using histograms.

    :param list_files: list of csv paths to read.
    :param list_names: list of names to associate with each file inside the plots.
    :param prop: string defining which property to evaluate. It can be [logP, MW, QED, SA, NP]. default_value='log_p'
    :return: the comparison between the distributions plots.
    """
    list_dataframes = []
    for file_path in list_files:
        df = pd.read_csv(file_path)
        list_dataframes.append(df)
    # Compute the property for each molecule
    scores = [list(df[prop]) for df in list_dataframes]
    # Plot the distributions
    sns.set()
    # Plot a distribution for each given txt file
    sns.histplot(data=scores, legend=False, multiple='dodge')
    # Insert the property name as x_label
    plt.xlabel(prop, fontsize=13)
    # Insert a legend to associate each file with its distribution plot
    plt.legend(loc='best', labels=list_names)
    plt.tight_layout()
    plt.show()


def plot_histograms(dataframe, figsize=(12, 4)):
    """
    Displays the bar plot related to a specific dataframe.

    :param dataframe: pandas dataframe containing the data.
    :param figsize: size of the figure to plot.
    :return: the image obtained.
    """
    sns.set()
    dataframe.plot.bar(rot=0, figsize=figsize)
    plt.tight_layout()
    plt.show()

