# Import packages
from tensorflow.keras.layers import Dense, Input
from spektral.layers import ECCConv, GlobalSumPool
from tensorflow.keras.models import Model
from numpy import random, array_split, triu, multiply, round, zeros, ones, array
import tensorflow as tf
from spektral.data.graph import Graph
from scipy.sparse import csr_matrix
from time import time
from Modules.visualization import *
import numpy as np


class SimpleGan:
    def __init__(self, embedding_dim, node_features, edge_features, max_nodes, atom_types, edge_types=4):
        """
        Creates a GAN model for de novo drug design.

        :param embedding_dim: size of the vector sampled from the normal distribution and fed as input to the generator.
        :param node_features: number of node features. Typically a one hot encoding vector describing the atom type.
        :param edge_features: number of edge features. Typically a one hot encoding vector describing the edge type.
        :param max_nodes: number of maximum nodes constituting the molecules processed.
        :param atom_types: number of atom types supported.
        :param edge_types: number of edge types supported. default_value=4
        """
        # Save important model parameters
        self.embedding_dim = embedding_dim
        self.node_features = node_features
        self.edge_features = edge_features
        self.max_nodes = max_nodes
        self.atom_types = atom_types
        self.edge_types = edge_types
        # Number of generator outputs dedicated to the A and X matrix respectively
        self.adjacency_matrix_outputs = int((self.max_nodes ** 2 - self.max_nodes) / 2)
        self.atom_feature_matrix_outputs = self.max_nodes
        # Instantiate the generator and the discriminator
        self.inputs = Input(shape=self.embedding_dim)
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

    def train(self, training_dataset):
        """
        Trains the GAN model.

        :param training_dataset: batch loader of the dataset.
        :return:
        """
        # This method returns a helper function to compute cross entropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # Discriminator and Generator optimizers
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        def train_step(real_batch):
            # Generate randomly the generator inputs sampling a normal distribution
            noise = tf.random.normal([batch_size, self.embedding_dim])
            # Record operations for automatic differentiation.
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Obtain generator outputs and organised them in a dedicated batch
                generated_molecules = self.generator(noise, training=True)
                generated_batch = self.prepare_batch(generated_molecules, batch_size)
                # Get the discriminator outputs with real and generated batches as inputs
                real_output = self.discriminator(real_batch, training=True)
                fake_output = self.discriminator(generated_batch, training=True)
                # Compute generator loss
                gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
                # compute discriminator loss
                real_loss = cross_entropy(tf.ones_like(real_output), real_output)
                fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
                disc_loss = real_loss + fake_loss
            # Compute the gradient using operations recorded in context of the generator tape.
            gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            # Compute the gradient using operations recorded in context of the discriminator tape.
            disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            # Update generator and discriminator parameters
            generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # Number of batches in every epoch
        steps_per_epoch = training_dataset.steps_per_epoch
        # Batch size
        batch_size = training_dataset.batch_size
        # Start the stopwatch to know the execution time of each epoch
        start = time()
        # Iterate over all the batches (num_batches*epochs) prepared in the training dataset
        for count, batch in enumerate(training_dataset):
            # Perform a training step
            train_step(batch)
            # After every epoch display the related information
            if count % steps_per_epoch == 0:
                print(f'Time for epoch {int(count / steps_per_epoch)} is {time() - start} sec')
                # Restart the stopwatch to know the execution time of each epoch
                start = time()

    def test(self, samples):
        """
        Test the generative model.

        :param samples: number of samples to generate.
        :return:
        """
        # Generate randomly the generator inputs sampling a normal distribution
        noise = random.normal(0, 1, size=[samples, self.embedding_dim])
        # Obtain generator outputs and organised them in a dedicated batch
        generated_molecules = self.generator(noise)
        # Transform each output in a Spektral Graph object
        generated_molecules = [self.output_to_graph(molecule) for molecule in generated_molecules]
        # Iterate over all oututs
        for molecule in generated_molecules:
            # Transform every Graph object into a rdkit mol object
            nodes, a = qm9_to_rdkit_notation(molecule)
            mol, validity = mol_from_graph(nodes, a)
            # If the molecule is valid...
            if validity:
                # Remove the non necessary separate atoms or molecules
                mol = single_molecule(mol)
                # Show the obtained molecule
                show_mol(mol)

    def create_generator(self):
        """
        Create a discriminator using Keras layers.

        :return: the generator model.
        """
        # The first generator layer operates on the inputs of the GAN model
        x = Dense(units=64, activation='relu')(self.inputs)
        x = Dense(units=96, activation='relu')(x)
        last_layer_units = self.adjacency_matrix_outputs + self.atom_feature_matrix_outputs
        outputs = Dense(units=last_layer_units, activation='sigmoid')(x)
        # Create a Keras Model object
        generator = Model(inputs=self.inputs, outputs=outputs, name='Generator')
        generator.summary()
        return generator

    def create_discriminator(self):
        """
        Create a discriminator using Spektral layers.

        :return: the discriminator model.
        """
        # The first discriminator layer operates on the outputs of the generator model.
        x_in = Input(shape=(self.node_features,), name='X_in')
        a_in = Input(shape=(None,), sparse=True, name='A_in')
        e_in = Input(shape=(self.edge_features,), name='E_in')
        i_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)
        # A series of Graph Convolutional Layers
        x_1 = ECCConv(32, activation='relu')([x_in, a_in, e_in])
        x_2 = ECCConv(32, activation='relu')([x_1, a_in, e_in])
        # Aggregate node information with a Graph Pooling Layer
        x_3 = GlobalSumPool()([x_2, i_in])
        outputs = Dense(units=1, activation='sigmoid')(x_3)
        # Create a Keras Model object
        discriminator = Model(inputs=[x_in, a_in, e_in, i_in], outputs=outputs)
        discriminator.summary()
        return discriminator

    def tf_output_to_graph(self, gen_outputs):
        """
        Converts each output of the generator into a graph represented by a node_matrix, a adj_matrix and a edge_matrix.

        :param gen_outputs: a single generator output.
        :return: the related matrices.
        """
        # The first part of an output array is dedicated to the upper triangle of the adjacency matrix
        adjacency = gen_outputs[:self.adjacency_matrix_outputs]
        # The remaining part describes the atom features.
        node_matrix = gen_outputs[self.adjacency_matrix_outputs:]
        # Map the range of values from [0,1] to [0, number of edge types] in I. 0 means no edges.
        adjacency = tf.cast(tf.round(tf.multiply(adjacency, self.edge_types)), dtype=tf.int64)
        # Create a mask to place the vector values within the upper triangle
        mask_indexes = tf.where(tf.constant(np.triu(np.ones((self.max_nodes, self.max_nodes), dtype=bool), k=1)))
        # Sparsely represent the adjacency matrix
        adj_matrix = tf.SparseTensor(mask_indexes, adjacency,
                                     dense_shape=tf.cast((self.max_nodes, self.max_nodes), dtype=tf.int64))
        # Convert the adjacency matrix to the dense representation and back to the sparse one. This allows to correct
        # the original sparse object that keeps the position also of the zero elements.
        adj_matrix = tf.sparse.to_dense(adj_matrix)
        adj_matrix = tf.sparse.from_dense(adj_matrix)
        # Copy the upper triangle in the lower triangle so that the matrix is symmetric
        adj_matrix = tf.sparse.add(adj_matrix, tf.sparse.transpose(adj_matrix))
        # Ont-hot encoding representation of the edges
        edge_matrix = tf.one_hot(adj_matrix.values - 1, self.edge_types)
        # Set all the non zero values of the adjacency matrix to ones
        adj_matrix = tf.SparseTensor(adj_matrix.indices, tf.ones(adj_matrix.indices.shape[0]),
                                     dense_shape=tf.cast((self.max_nodes, self.max_nodes), dtype=tf.int64))
        # Convert the range [0,1] to [0, num_atom_types - 1] in I.
        node_matrix = tf.cast(tf.round(tf.multiply(node_matrix, self.atom_types - 1)), dtype=tf.int64)
        # Ont-hot encoding representation of the nodes
        node_matrix = tf.one_hot(node_matrix, self.atom_types)
        return node_matrix, tf.sparse.to_dense(adj_matrix), edge_matrix

    def prepare_batch(self, generated_molecules, batch_size):
        """
        Translate the generator outputs in a disjoint batch so that they can be processed by the Spektral layers. The
        entire function is implemented using only tensorflow.

        :param generated_molecules: generator outputs as array or list of arrays (generator output Tensor).
        :param batch_size: dimension of the batch, i.e. number of samples included.
        :return: a disjoint batch of the generated molecules.
        """
        # For every generated molecule get the node, adjacency and edge matrices
        node_matrix, adj_matrix, edge_matrix = tf.map_fn(fn=lambda t: self.tf_output_to_graph(t),
                                                         elems=generated_molecules,
                                                         fn_output_signature=(tf.float32, tf.float32, tf.float32))
        # Separate and concatenate all the node matrices obtained
        x_list = tf.unstack(node_matrix)
        node_matrix = tf.concat(x_list, axis=0)
        # Separate and concatenate all the edge matrices obtained
        e_list = tf.unstack(edge_matrix)
        edge_matrix = tf.concat(e_list, axis=0)
        # Keep notes for each tensors of the related molecule
        i_out = tf.repeat(tf.range(start=0, limit=len(x_list), delta=1), [x_list[0].shape[0]] * batch_size)
        # Get a list of the adjacency matrices obtained
        blocks = tf.unstack(adj_matrix)
        # Cast every matrix into a Linear Operator
        lin_op_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in blocks]
        # Combines one or more LinearOperators in to a Block Diagonal matrix
        lin_op_block_diagonal = tf.linalg.LinearOperatorBlockDiag(lin_op_blocks)
        # Get the sparse representation of the Block Diagonal matrix
        adj_matrix = tf.sparse.from_dense(lin_op_block_diagonal.to_dense())
        return node_matrix, adj_matrix, edge_matrix, i_out

    def output_to_graph(self, gen_outputs):
        """
        Translate the generator outputs in graphs so that they can be processed by the Spektral layers.

        :param gen_outputs: generator outputs as array or list of arrays (nd array).
        :return:
        """
        # The first part of an output array is dedicated to the upper triangle of the adjacency matrix. The remaining
        # part describes the atom features.
        adjacency, node_matrix = array_split(gen_outputs, [self.adjacency_matrix_outputs])
        # Map the range of values from [0,1] to [0, number of edge types]. 0 means no edges.
        adjacency = multiply(adjacency, self.edge_types)
        # Round each value to the nearest integer
        adjacency = round(adjacency)
        # Allocate the adjacency matrix and set the upper triangle with the adjacency values
        # The adjacency matrix has a fixed size equal to max_nodes x max_nodes
        adj_matrix = zeros((self.max_nodes, self.max_nodes), dtype='int')
        adj_matrix[triu(ones((self.max_nodes, self.max_nodes), dtype=bool), k=1)] = adjacency
        # Render the adjacency matrix symmetric
        adj_matrix = adj_matrix + adj_matrix.T
        # Get the adjacency matrix sparse representation
        adj_matrix = csr_matrix(adj_matrix)
        # Create the one-hot encoding representation of all the existing edges
        edge = []
        for row, col in zip(*adj_matrix.nonzero()):
            # Take note of the edge type
            edge.append(adj_matrix[row, col] - 1)
            # Set the controlled value to 1
            adj_matrix[row, col] = 1
        # Ont-hot encoding representation of the edges
        edge_matrix = tf.one_hot(edge, self.edge_types).numpy()
        # Convert the range [0,1] to [0, num_atom_types - 1]
        node_matrix = multiply(node_matrix, self.atom_types - 1)
        # Round each value to the nearest integer
        node_matrix = round(node_matrix)
        # Ont-hot encoding representation of the atoms
        node_matrix = tf.one_hot(node_matrix, self.atom_types).numpy()
        # Return the related Spektral Graph object
        return Graph(x=node_matrix, a=adj_matrix, e=edge_matrix, y=None)
