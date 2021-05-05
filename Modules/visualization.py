# Import packages
from rdkit.Chem import MolToSmiles, Draw
from Modules.utils import *
import matplotlib.pyplot as plt


def show_mol(mol, name=False):
    """
    Displays a rdkit mol object

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





