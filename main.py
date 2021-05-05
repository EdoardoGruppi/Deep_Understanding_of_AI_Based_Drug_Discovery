# Import packages
from Modules.data_preparation import *
from Models.first_model import *

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