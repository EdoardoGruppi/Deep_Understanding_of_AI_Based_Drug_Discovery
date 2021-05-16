# Import packages
from tensorflow.keras.layers import Dense, Input, Conv1D, GRU, RepeatVector, Flatten, TimeDistributed
from tensorflow.keras.models import load_model, Model
import os
from Modules.components import Sampling
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
from math import ceil
from Modules.smiles_data_preparation import gen_data_vae
from numpy.random import normal


class SmilesVae:
    def __init__(self, charset, max_length, latent_rep_size=292):
        """
        Creates an instance of the SmilesVAE model.

        :param charset: chars to integers dictionary.
        :param max_length: maximum sequence length.
        :param latent_rep_size: size of the latent vector produced by the encoder.
        """
        # Chars to integers dictionary and its length
        self.charset = charset
        self.charset_length = len(charset)
        # Maximum sequence length
        self.max_length = max_length
        # Size of the latent vector produced by the encoder
        self.latent_rep_size = latent_rep_size
        # Integers to chars dictionary
        self.int_to_char = dict(zip(charset.values(), charset.keys()))

        # Build the encoder model
        self.encoder = self.create_encoder(epsilon_std=0.01)
        self.encoder.summary()
        # Build the decoder model
        self.decoder = self.create_decoder()
        self.decoder.summary()
        # Build the Variational AutoEncoder model
        inputs = Input(shape=(self.max_length, self.charset_length))
        z_latent, z_log_var, z_mean = self.encoder(inputs)
        outputs = self.decoder(z_latent)
        self.model = Model(inputs, outputs)
        # Add loss to the model and compile it
        self.model.add_loss(self.vae_loss(inputs, outputs, z_log_var, z_mean))
        self.model.compile(optimizer='Adam', loss=None, metrics=['accuracy'])
        self.model.summary()

    def train(self, train_data, epochs, batch_size, save=True):
        """
        Trains the model.

        :param train_data: training samples.
        :param batch_size: number of samples reserved to each batch.
        :param epochs: number of epochs to process.
        :param save: boolean. If True the model is saved. default_value=True
        :return:
        """
        # Fit the model
        history = self.model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size)
        # plot the history of the model
        plt.plot(history.history["loss"], '-o', label="Loss")
        plt.legend()
        plt.show()
        if save:
            # Store to not having to train again...
            self.model.save(os.path.join('Saved Models', 'SmilesVAE'))

    def test(self, test_data, batch_size):
        """
        Tests the model.

        :param test_data: testing samples.
        :param batch_size: number of samples reserved to each batch.
        :return:
        """
        # Evaluate the model on the test dataset
        metric = None
        # Divide the test dataset into batches and sequentially process them
        for i in range(ceil(len(test_data) / batch_size)):
            # Load current batch
            test_batch = test_data[batch_size * i:batch_size * (i + 1)]
            # Test the model on the current batch
            metric = self.model.test_on_batch(test_batch, test_batch, reset_metrics=False)
        print(f'Average accuracy on testing dataset: {metric[-1]}')

    def generate_molecules(self, samples, near_molecule=0, stddev=0.1, mean=0, filename='SmilesVAE Results'):
        """
        Generate new molecules and save them into a filename.txt file.

        :param samples: number of molecules to generate.
        :param stddev: standard deviation of the gaussian from which to sample the vector. default_value=0.1
        :param mean: mean of the gaussian from which to sample the vector. default_value=0
        :param near_molecule: smiles of a molecule whose latent vector is nears of the one sampled. default_value=0
        :param filename: name of the file on which to save the generated molecules. default_value='SmilesVAE Results'
        :return: the path directory to the filename.txt file.
        """
        # If a smiles molecule is given in input
        if near_molecule != 0:
            # Get the one hot encoding representation of the molecule
            one_hot = gen_data_vae(near_molecule, self.charset, self.max_length)
            # Obtain the latent vector related to the molecule
            near_molecule = self.encoder.predict(one_hot)
        # Sample the latent vectors from the distribution
        latent_vectors = normal(loc=mean, scale=stddev, size=(samples, self.latent_rep_size)) + near_molecule

        def predict_smiles(mol_one_hot):
            """
            Generate a new molecule from a one-hot encoding vector.

            :param mol_one_hot: one-hot encoding vector representing the molecule.
            :return: the smiles string of the generated molecule.
            """
            # Obtain the one-hot encoding vector of the generated molecule
            molecule = self.decoder.predict(mol_one_hot.reshape(1, self.latent_rep_size)).argmax(axis=2)[0]
            # Convert the one-hot representation into a smiles string
            molecule = "".join(map(lambda x: self.int_to_char[x], molecule)).strip()
            # Return the smiles string without the stop characters
            return molecule.split('E')[0]

        # Generate new molecules
        new_molecules = [predict_smiles(vector) for vector in latent_vectors]
        # Path to the filename.txt file
        results_file = os.path.join('./Saved Results', filename)
        # Write all the molecules in the file, one per line
        with open(results_file, 'w') as f:
            for mol in new_molecules:
                # Delete the first and last characters also
                f.write(f'{mol[1:-1]}\n')
        return results_file

    def load_model(self):
        """
        Load a model previously saved.
        """
        # Load to continue training or evaluate...
        self.model = load_model(os.path.join('Saved Models', 'SmilesVAE'))

    def create_encoder(self, epsilon_std):
        """
        Creates the encoder model.

        :param epsilon_std: standard deviation of the normalized form which part of the latent vector is sampled.
        :return: the encoder model.
        """
        # The first part of the model is composed by three convolutional layers and a dense layer
        inputs = Input(shape=(self.max_length, self.charset_length))
        h = Conv1D(filters=9, kernel_size=9, activation='relu', name='Conv_1')(inputs)
        h = Conv1D(filters=9, kernel_size=9, activation='relu', name='Conv_2')(h)
        h = Conv1D(filters=10, kernel_size=11, activation='relu', name='Conv_3')(h)
        h = Flatten(name='Flatten_1')(h)
        h = Dense(units=435, activation='relu', name='Dense_1')(h)
        # Dense layer returning the mean vector of the latent distribution
        z_mean = Dense(units=self.latent_rep_size, activation='linear', name='Z_mean')(h)
        # Dense layer returning the log variance vector of the latent distribution
        z_log_var = Dense(units=self.latent_rep_size, activation='linear', name='Z_log_var')(h)
        # Custom layer returning the latent vector
        z_vector = Sampling(latent_rep_size=self.latent_rep_size, epsilon_std=epsilon_std)([z_mean, z_log_var])
        return Model(inputs, [z_vector, z_log_var, z_mean])

    def create_decoder(self):
        """
        Creates the decoder model.

        :return: the decoder model.
        """
        # The model is composed by one Dense layer followed by three GRU layers
        encoded_input = Input(shape=(self.latent_rep_size,))
        h = Dense(self.latent_rep_size, activation='relu', name='Latent_input')(encoded_input)
        # Repeat the input n times
        h = RepeatVector(n=self.max_length, name='Repeat_vector')(h)
        h = GRU(501, return_sequences=True, name='Gru_1')(h)
        h = GRU(501, return_sequences=True, name='Gru_2')(h)
        h = GRU(501, return_sequences=True, name='Gru_3')(h)
        # Wrapper to apply a layer to every temporal slice of an input
        outputs = TimeDistributed(Dense(self.charset_length, activation='softmax'), name='Decoded_mean')(h)
        return Model(encoded_input, outputs)

    def vae_loss(self, output, output_predicted, z_log_var, z_mean):
        """
        Loss function used to update the model.

        :param output: true outputs.
        :param output_predicted: predicted outputs.
        :param z_log_var: the computed log variance vector of the latent distribution.
        :param z_mean: the computed mean vector of the latent distribution.
        :return: the VAE loss
        """
        # Compute the Kullback–Leibler loss
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        output = K.flatten(output)
        output_predicted = K.flatten(output_predicted)
        # Compute the binary crossentropy loss
        cross_loss = self.max_length * binary_crossentropy(output, output_predicted)
        return cross_loss + kl_loss
