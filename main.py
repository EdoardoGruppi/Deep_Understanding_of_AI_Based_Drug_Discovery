# Import packages
from Modules.graph_data_preparation import *
from Models.first_model import *
from Models.simple_graph_gan import *
from Modules.smiles_data_preparation import *
from Models.char_rnn import CharRNN
from Models.smiles_vae import SmilesVae
import tensorflow as tf
import os

# # set_memory_growth() allocates exclusively the GPU memory needed
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# if len(physical_devices) is not 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# # Load Dataset
# dataset, node_features, edge_features, n_out = load_qm9(amount=96)
# # Train/test split
# loader_tr, loader_te = split_dataset(dataset, batch_size=32, epochs=10, train_percentage=0.9)
# # Build model
# model = FirstModel(node_features, edge_features)
# # Fit model
# model.train(loader_tr, learning_rate=1e-3)
# # Test model
# model.test(loader_te)

# SimpleGAN ========================================================
# # Load Simple Gan
# model = SimpleGan(embedding_dim=10, node_features=qm9_atoms_number, edge_features=4, max_nodes=15,
#                   atom_types=qm9_atoms_number, edge_types=4)
# # Load training dataset
# dataset = QM9(amount=50)
# train_dataset = load_dataset(dataset, node_features=5, batch_size=5, epochs=10, edge_features=4)
# # Train the Gan model
# model.train(training_dataset=train_dataset)
# # Test the Gan model
# model.test(samples=5)

# # CharRNN =========================================================
# # Load the data
# train_data, test_data, train_labels, test_labels, char_to_int, int_to_char = load_datasets('CHEMBL_FULL_DATASET.txt',
#                                                                                            test_size=0.1,
#                                                                                            samples=5000)
# # Instantiate the model
# model = CharRNN(input_shape=train_data.shape[2], output_shape=train_labels.shape[-1], char_int_dict=char_to_int,
#                 int_char_dict=int_to_char)
# # Train the model
# model.train(train_data, train_labels, batch_size=128, epochs=2, save=True)
# # Test the model
# model.test(test_data, test_labels, batch_size=128)
# # # Load the model
# # model.load_model()
# # Generate new molecules
# model.generate_molecules(new_molecules=50)
#
# # SmilesVAE ========================================================
# # Load the data
# train_data, test_data, charset, length = load_datasets_smiles_vae('CHEMBL_FULL_DATASET.txt', test_size=0.1,
#                                                                   samples=5000)
# # Instantiate the model
# model = SmilesVae(charset=charset, max_length=length, latent_rep_size=292)
# # Train the model
# model.train(train_data, batch_size=128, epochs=2, save=True)
# # Test the model
# model.test(test_data, batch_size=128)
# # # Load the model
# # model.load_model()
# # Generate new molecules
# model.generate_molecules(samples=50, near_molecule=0, stddev=0.1, mean=0)

# Additional =========================================================

# create_cond_dataset('CHEMBL_FULL_DATASET.txt', 'chembl_prop.txt', activity=False)
# dataset_activity(dataset_file=os.path.join(base_dir, 'CHEMBL_FULL_DATASET.txt'), targets=['CHEMBL262'])
# activity_subset('chembl_CFD_activity.csv', activity=['active'], confidence=90)
# create_cond_dataset('active_chembl_27.txt', 'active_chembl_27_prop.txt', activity=False)
create_cond_dataset('active_chembl_27.txt', 'active_chembl_27_prop_gsk.txt', activity=True)

# extract_smiles_from_gcpn_csv(filename='molecule_zinc_test_conditional_4200.csv', num_samples=5000)

# test_dataset = os.path.join(base_dir, 'GCPN_molecules.txt')
# molecules_passed = filter_molecules(test_dataset, check='complete', filters='pains_filters.txt',
#                                     qed=0.5, sa=5)
