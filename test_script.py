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
# .txt file that includes the SMILES strings of the training dataset
reference_dataset_filename = 'chembl_no_actives.txt'
# Pickle file containing the fingerprints of the reference dataset
fingerprints_file = 'Datasets/chembl_no_actives_fingerprints.pkl'
# Filename of the .txt file of the generated molecules, one smiles string per line
test_dataset_filename = 'new_ck_ppara_molecules_2.txt'
save_filename = test_dataset_filename.replace('molecules', 'metrics')
# Number of training samples involved in the computation of the FCD and KL metrics
subset_size = 2000
# ------------------------------------------------------------------

# Create directory if it does not already exist
os.makedirs(savings_dir, exist_ok=True)
# Open the txt file to write all the results achieved
file = open(os.path.join(savings_dir, save_filename), "w")
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
uniqueness_value = uniqueness(test_file=test_dataset)
novelty_value = novelty(train_file=ref_dataset, test_file=test_dataset)
file.flush()

# Comparison between the property distributions
for prop in ['logP', 'MW', 'QED', 'SA', 'NP', 'Atoms']:
    property_distributions([ref_dataset, test_dataset],
                           list_names=['Reference', 'Generated'], prop=prop, txt=True)

# Compute the FCD score
FCD_score, fcd_score = frechet_distance(ref_dataset, test_dataset, sample_size=subset_size)
file.flush()

# Compute the KL divergence value
kl_score = kl_divergence(ref_dataset, test_dataset, subset_size=subset_size)
file.flush()

# Compute SNN and DfD metrics of the generated molecules
snn, dfd = snn_metric(fingerprints_file, test_dataset, pkl=True, batch_size=500)
file.flush()

# Compute Internal Diversity of the generated molecules
int_div = internal_diversity(test_dataset, p=1)
file.flush()

# Filtering generated molecules
_ = filter_molecules(test_dataset, check='complete', filters='pains_filters.txt')
file.flush()

# Filtering with QED and SA scores
print('Filtering also with selected QED and SA scores.')
_ = filter_molecules(test_dataset, check='complete', filters='pains_filters.txt', qed=0.5, sa=5)
file.flush()

# Compute the atom occurrences
atom_occurrences(test_dataset, atoms_list=chembl_atoms_list)
file.flush()

# Get activity
print('Activity computed by means of the ChEMBL27 model')
# BACE1 'CHEMBL4822' - PPARalpha 'CHEMBL239' - CDK2 'CHEMBL301' - DRD3 'CHEMBL234'
results = get_activity(test_dataset, confidence=80, targets=['CHEMBL239'], activity=['active'])
file.flush()

# Close the file
file.close()
