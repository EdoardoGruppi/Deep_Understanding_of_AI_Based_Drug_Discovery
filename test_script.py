"""
Script to compute all the metrics available on some provided datasets.

"""
# Import packages
from Modules.metrics import *
from Modules.config import *
import os
import sys

# USER INPUTS -----------------------------------------------------
# Path to the directory where the dataset are saved
datasets_dir = base_dir
# Path to the directory where the results are saved
savings_dir = os.path.join('Savings', 'Saved Results')
# Filenames of the generated .txt files, one smiles string per line
reference_dataset_filename = 'CHEMBL_FULL_DATASET.txt'
test_dataset_filename = 'molRNN_molecules.txt'
save_filename = 'molRNN_metrics.txt'
subset_size = 2000
# ------------------------------------------------------------------

# Savings directory path
directory = os.path.join('Savings', 'Saved Results')
# Create directory if it does not already exist
os.makedirs(directory, exist_ok=True)
# Open the txt file to write all the results achieved
file = open(os.path.join(directory, save_filename), "w")
sys.stdout = file

# Path to the datasets
ref_dataset = os.path.join(datasets_dir, reference_dataset_filename)
test_dataset = os.path.join(datasets_dir, test_dataset_filename)
print('USER INPUTS ' + '-' * 30)
print(f'Reference dataset in {ref_dataset}')
print(f'Test dataset in {test_dataset}')
print('-' * 42 + '\n')
file.flush()

# Validity check
print('Validity test on the datasets provided...')
# ratio1, _ = validity(ref_dataset)
ratio2, _ = validity(test_dataset)
# print(f'The {ratio1 * 100:5.2f} % of molecules in the reference dataset are valid.')
print(f'The {ratio2 * 100:5.2f} % of molecules in the test dataset are valid.\n')
file.flush()

# Novelty and Uniqueness computation
novelty_value = novelty(train_file=ref_dataset, test_file=test_dataset)
uniqueness_value = uniqueness(test_file=test_dataset)
file.flush()

# # Comparison between the property distributions
# for prop in ['logP', 'MW', 'QED', 'SA', 'NP', 'Atoms']:
#     property_distributions([
#         os.path.join(base_dir, 'CHEMBL_FULL_DATASET.txt'),
#         os.path.join(base_dir, 'HierVAE_molecules.txt'),
#         os.path.join(base_dir, 'Yasonik_molecules.txt')
#         ],
#         list_names=['CHEMBL', 'HierVAE', 'Yasonik'], prop=prop, txt=True)

# Compute the FCD score
FCD_score, fcd_score = frechet_distance(ref_dataset, test_dataset, sample_size=subset_size)
file.flush()

# Filtering generated molecules
_ = filter_molecules(test_dataset, check='complete', filters='pains_filters.txt')
file.flush()

# Filtering with QED and SA scores
print('Filtering also with selected QED and SA scores.')
_ = filter_molecules(test_dataset, check='complete', filters='pains_filters.txt', qed=0.5, sa=5)
file.flush()

# Compute Internal Diversity of the generated molecules
int_div = internal_diversity(test_dataset, p=1)
file.flush()

# Compute SNN and DfD metrics of the generated molecules
snn, dfd = snn_metric(ref_dataset, test_dataset)
file.flush()

# Compute the atom occurrences
atom_occurrences(test_dataset, atoms_list=chembl_atoms_list)
file.flush()

# Compute the KL divergence value
kl_score = kl_divergence(ref_dataset, test_dataset, subset_size=subset_size)
file.flush()

# Get activity
results = get_activity(test_dataset, confidence=90, targets=['CHEMBL262'], activity=['active'])
file.flush()

# Close the file
file.close()
