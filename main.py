# Import packages
from Modules.graph_data_preparation import *
from Models.first_model import *
from Models.simple_graph_gan import *
from Modules.smiles_data_preparation import load_datasets
from Models.char_rnn import CharRNN

# Load Dataset
dataset, node_features, edge_features, n_out = load_qm9(amount=96)
# Train/test split
loader_tr, loader_te = split_dataset(dataset, batch_size=32, epochs=10, train_percentage=0.9)
# Build model
model = FirstModel(node_features, edge_features)
# Fit model
model.train(loader_tr, learning_rate=1e-3)
# Test model
model.test(loader_te)

# SimpleGAN ========================================================
# Load Simple Gan
model = SimpleGan(embedding_dim=10, node_features=qm9_atoms_number, edge_features=4, max_nodes=15,
                  atom_types=qm9_atoms_number, edge_types=4)
# Load training dataset
dataset = QM9(amount=1000)
train_dataset = load_dataset(dataset, node_features=5, batch_size=5, epochs=10, edge_features=4)
# Train the Gan model
model.train(training_dataset=train_dataset)
# Test the Gan model
model.test(samples=5)

# CharRNN =========================================================
# Load the data
train_data, test_data, train_labels, test_labels, char_to_int, int_to_char = load_datasets('CHEMBL_FULL_DATASET.txt',
                                                                                           test_size=0.1, samples=5000)
# Instantiate the model
model = CharRNN(input_shape=train_data.shape[2], output_shape=train_labels.shape[-1], char_int_dict=char_to_int,
                int_char_dict=int_to_char)
# Train the model
model.train(train_data, train_labels, batch_size=128, epochs=2, save=True)
# Test the model
model.test(test_data, test_labels, batch_size=128)
# # Load the model
# model.load_char_rnn()
# Generate new molecules
model.generate_molecules(new_molecules=50)
