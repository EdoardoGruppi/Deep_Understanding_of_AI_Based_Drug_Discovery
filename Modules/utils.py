# Import packages
import requests
from rdkit.Chem.rdchem import BondType, RWMol, Atom
import os
from rdkit.Chem import SanitizeMol, MolToSmiles, MolFromSmiles
from Modules.config import *
from pandas import read_csv

# Dictionary to identify the type of each bond
bond_types = {1: BondType.SINGLE, 2: BondType.DOUBLE, 3: BondType.TRIPLE, 4: BondType.AROMATIC}
# Dictionary to translate the atom types from qm9 notation to that of rdkit.
# In particular in qm9 the H, C, N, O, F are defined as 0, 1, 2, 3, 4.
atom_types_qm9 = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}


def mol_from_graph(nodes, adjacency_matrix):
    """
    Converts a given graph represented by the list of nodes and the adjacency matrix to a rdkit mol object. Note: the
    result may differ from the given molecule since the last statement adds hydrogen atoms where needed to
    respect valency. The mol object outcome may comprise distinct molecules. Use the single_molecule function to
    retain only the greatest one of them.

    :param nodes: list of nodes related to the adjacency matrix.
    :param adjacency_matrix: matrix to define the connections if any within the molecule.
    :return: the rdkit mol object related to the input graph.
    """
    # Create empty editable mol object
    mol = RWMol()
    node_to_idx = {}
    # Add the nodes and keep track of the insertion order
    for i in range(len(nodes)):
        molIdx = mol.AddAtom(Atom(nodes[i]))
        node_to_idx[i] = molIdx
    # Add bonds as expressed in the A matrix
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            # Read only a half of A (symmetric)
            if (iy < ix) & (bond > 0):
                # If there is a bond select the type and connect the correct two atoms
                bond_type = bond_types[bond]
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
    # Convert RWMol to Mol object
    mol = mol.GetMol()
    # Kekulize, check valencies, set aromaticity, conjugation and hybridization. If the atom valencies are violated
    # the molecule is considered invalid.
    validity = try_except(lambda: SanitizeMol(mol), ValueError)
    return mol, validity


def qm9_to_rdkit_notation(mol):
    """
    Converts the notation with which a molecule is described from th qm9 notation to the one of rdkit.

    :param mol: molecule defined with the qm9 notation.
    :return: the list of nodes and the adjacency matrix under the rdkit notation.
    """
    # Find the atom corresponding to each node. Originally it is expressed as a one-hot encoding vector
    nodes = mol.x[:, 0:5].argmax(axis=1)
    # Convert the qm9 notation to the rdkit notation for every node
    nodes = [atom_types_qm9[node] for node in nodes]
    # Find the edge types of the edges
    edge_types = mol.e.argmax(axis=1) + 1
    # Iterate all the non zero element of the matrix and change its value to the corresponding edge type
    for count, (row, col) in enumerate(zip(*mol.a.nonzero())):
        mol.a[row, col] = edge_types[count]
    # Convert the sparse representation to a full adjacency matrix
    adjacency_matrix = mol.a.toarray()
    return nodes, adjacency_matrix


def smiles_to_name(smiles):
    """
    Finds the name of a molecule expressed as SMILES string

    :param smiles: smiles string to identify.
    :return: the name of the molecule given as input.
    """
    # Url of the website on which to perform the research
    url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name"
    # Send an http get request
    response = requests.get(url)
    # If the request is successful return the name
    if 'text/plain' in response.headers['Content-Type']:
        name = response.text
    else:
        name = 'Name not found'
    return name


def single_molecule(mol):
    """
    If a molecule object comprises two distinct molecules, this function keeps only the greatest one.

    :param mol: molecule object to evaluate.
    :return: the new molecule object with a sole molecule.
    """
    # Convert the mol object to a smiles string
    smiles = MolToSmiles(mol)
    # If a dot is within the string the mol object contains two separate molecules
    if '.' in smiles:
        # Split the smiles string whenever a dot is met and keep the longest segment.
        smiles = max(smiles.split('.'), key=len)
        # Convert the string back into a mol object
        mol = MolFromSmiles(smiles)
    return mol


def extract_smiles_from_gcpn_csv(filename, num_samples):
    """


    :param filename:
    :param num_samples:
    :return:
    """
    dataframe = read_csv(os.path.join(base_dir, filename), sep=',')
    smiles = [item[0] for item in dataframe.index.values]
    with open(os.path.join(base_dir, 'GCPN_results.txt'), 'w') as f:
        count = 0
        exceptions = 0
        while count < num_samples:
            smiles_string = smiles[-(1 + count + exceptions)]
            if '****' not in smiles_string:
                f.write(smiles_string + '\n')
                count += 1
            else:
                exceptions += 1
                print(f'Actual count: {count}')
                print(smiles_string)
