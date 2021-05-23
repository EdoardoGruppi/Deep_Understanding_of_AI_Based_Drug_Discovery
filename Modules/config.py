# File containing all the global variables
# It includes also some useful function non strictly related to the project itself

# Relative path to the datasets directory
base_dir = './Datasets'
# Number of atom types in the QM9 dataset
qm9_atoms_number = 5
# List of atoms in the CHEMBL dataset. The list is obtained with the get_atom_list function in Modules.metrics.py
# get_atom_list(file_path_chembl_full_dataset)
chembl_atoms_list = ['F', 'I', 'O', 'C', 'S', 'Br', 'Cl', 'P', 'N']


def try_except(function, expected_exc, result=True, default=False):
    """
    Try and Except statement in a single line.

    :param function: function to try.
    :param expected_exc: exception to monitor.
    :param result: result returned if no error occurs. default_value=True
    :param default: result returned if an error occurs. default_value=False
    :return:
    """
    try:
        function()
        return result
    except expected_exc:
        return default
