# Import packages
from rdkit.Chem import Draw, GetAdjacencyMatrix
from Modules.utils import *
import matplotlib.pyplot as plt
from Modules.metrics import *


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

