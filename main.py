# Import packages
from Modules.data_preparation import *
from Models.first_model import *
from Models.simple_graph_gan import *

# Load Dataset
dataset, node_features, edge_features, n_out = load_qm9(amount=96)
# Train/test split
loader_tr, loader_te = split_dataset(dataset, batch_size=32, epochs=10, train_percentage=0.9)
# Build model
model = FirstModel(node_features, edge_features, n_out)
# Fit model
model.train(loader_tr, learning_rate=1e-3)
# Test model
model.test(loader_te)

# ========================================================
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
