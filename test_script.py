"""
Script to compute all the metrics available on some provided datasets.

"""
# Import packages
from Modules.metrics import *
from Modules.config import *
import os
import sys

# USER INPUTS -----------------------------------------------------
# Path to the directory where the datasets are saved
datasets_dir = base_dir
# Filenames of the datasets .txt files, one smiles string per line
reference_dataset_filename = 'example.txt'
test_dataset_filename = 'example2.txt'
save_filename = 'output.txt'
# ------------------------------------------------------------------

# Open the txt file to write all the results achieved
file = open("output.txt", "w")
sys.stdout = file

# Path to the datasets
ref_dataset = os.path.join(datasets_dir, reference_dataset_filename)
test_dataset = os.path.join(datasets_dir, test_dataset_filename)
print('USER INPUTS ' + '-' * 30)
print(f'Reference dataset in {ref_dataset}')
print(f'Test dataset in {test_dataset}')
print('-' * 42 + '\n')

# Validity check
print('Validity test on the datasets provided...')
ratio1, _ = validity(ref_dataset)
ratio2, _ = validity(test_dataset)
print(f'The {ratio1 * 100:5.2f} % of molecules in the reference dataset are valid.')
print(f'The {ratio2 * 100:5.2f} % of molecules in the test dataset are valid.\n')

# Novelty and Uniqueness computation
novelty_value = novelty(train_file=ref_dataset, test_file=test_dataset)
uniqueness_value = uniqueness(test_file=test_dataset)

# Comparison between the property distributions
for prop in ['logP', 'MW', 'QED', 'SA', 'NP']:
    property_distributions([ref_dataset, test_dataset], list_names=['Reference', 'Generated'], prop=prop, txt=True)

# Compute the FCD score
FCD_score, fcd_score = frechet_distance(ref_dataset, test_dataset)

# Filtering generated molecules
molecules_passed = filter_molecules(test_dataset, check='complete', filters='pains_filters.txt')

# Compute Internal Diversity of the generated molecules
int_div = internal_diversity(test_dataset, p=1)

# Compute SNN and DfD metrics of the generated molecules
snn, dfd = snn_metric(ref_dataset, test_dataset)

# Compute the atom occurrences
atom_occurrences(test_dataset, atoms_list=chembl_atoms_list)

# Compute the KL divergence value
kl_score = kl_divergence(ref_dataset, test_dataset)

# Close the file
file.close()
