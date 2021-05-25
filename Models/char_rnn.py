# Import packages
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential, load_model
import os
import matplotlib.pyplot as plt
from math import ceil
import numpy as np


# The model exploits a series of RNN layers to sequentially generate a new molecule. The input and the output
# molecules are represented by SMILES strings.


class CharRNN:
    def __init__(self, input_shape, output_shape, char_int_dict, int_char_dict):
        """
        Creates an instance of the CharRNN model.

        :param input_shape: shape of the input. Usually the number of unique chars.
        :param output_shape: shape of the output. Usually the number of unique chars.
        :param char_int_dict: dictionary to map chars to integers.
        :param int_char_dict: dictionary to map integers to chars.
        """
        self.char_int_dict = char_int_dict
        self.int_char_dict = int_char_dict
        # Build the model
        self.model = Sequential([
            LSTM(units=16, input_shape=(None, input_shape), return_sequences=True),
            Dropout(rate=0.25),
            LSTM(units=16, return_sequences=True),
            Dropout(rate=0.25),
            Dense(units=output_shape, activation='softmax')
        ])
        print(self.model.summary())
        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train(self, train_data, train_labels, batch_size, epochs, save=True):
        """
        Trains the model.

        :param train_data: training samples.
        :param train_labels: labels associated with the training samples.
        :param batch_size: number of samples reserved to each batch.
        :param epochs: number of epochs to process.
        :param save: boolean. If True the model is saved. default_value=True
        :return:
        """
        # Fit the model
        history = self.model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
        # plot the history of the model
        plt.plot(history.history["loss"], '-o', label="Loss")
        plt.legend()
        plt.show()
        if save:
            # Store to not having to train again...
            self.model.save(os.path.join('Saved Models', 'SimpleCharRNN'))

    def test(self, test_data, test_labels, batch_size):
        """
        Tests the model.

        :param test_data: testing samples.
        :param test_labels: labels associated with the testing samples.
        :param batch_size: number of samples reserved to each batch.
        :return:
        """
        predictions = []
        for i in range(ceil(len(test_data) / batch_size)):
            # Calculate predictions
            predictions.append(self.model.predict_on_batch(test_data[batch_size * i:batch_size * (i + 1)]))
        predictions = np.concatenate(predictions, axis=0)
        # Compare predictions to the correct results
        diff_results = np.ndarray.argmax(test_labels, axis=2) - np.ndarray.argmax(predictions, axis=2)
        # Count correct and incorrect predictions
        no_false = np.count_nonzero(diff_results)
        no_true = diff_results.size - no_false
        print(f'Average success rate on dataset: {np.round(100 * no_true / diff_results.size, 2)}')

    def load_model(self):
        """
        Load a model previously saved.
        """
        # Load to continue training or evaluate...
        self.model = load_model(os.path.join('Saved Models', 'SimpleCharRNN'))

    def generate_molecules(self, new_molecules, filename='CharRNN Results', plot=False):
        """
        Generate new molecules and save them into a filename.txt file.

        :param new_molecules: number of molecules to generate.
        :param filename: name of the file on which to save the generated molecules. default_value='CharRNN Results'
        :param plot: boolean. If True plots the predictions and the probabilities for each molecule. default_value=False
        :return: the path directory to the filename.txt file.
        """
        # Generate the required number of molecules
        new_molecules = [self.generate_single_molecule(plot=plot) for _ in range(new_molecules)]
        # Path to the filename.txt file
        results_file = os.path.join('./Saved Results', filename)
        # Write all the molecules in the file, one per line
        with open(results_file, 'w') as f:
            for molecule in new_molecules:
                # Delete the first and last characters also
                f.write(f'{molecule[1:-1]}\n')
        return results_file

    def generate_single_molecule(self, plot=False):
        """
        Generates a sole function.

        :param plot: boolean.  If True plots the predictions and the probabilities computed.
        :return: the string ('!smilesE') related to the new molecule. default_value=False
        """
        # Starting characters
        start_char = '!'
        # Array of probabilities
        predictions = []
        # Number of unique chars considering also ! and E
        dict_length = len(self.char_int_dict)
        # One-hot encoding representation of the first character
        x = np.zeros((1, 1, dict_length))
        x[0, 0, self.char_int_dict[start_char]] = 1
        # Instantiate current molecule state and current char
        molecule_string, char = start_char, start_char
        # Continue as long as 'E' is not produced
        while char != 'E':
            # Predict the probabilities for each char as the next element
            pred = self.model.predict(np.expand_dims(x[:, -1, :], axis=1)).flatten()
            # Randomly sample the new char while considering the predicted probabilities
            char = self.int_char_dict[np.random.choice(np.arange(0, dict_length), p=pred)]
            # Save the predicted probability values
            predictions.append(pred)
            # Update the current molecule state
            molecule_string += char
            # Update the one-hot encoding representation of the new molecule
            x1 = np.zeros((1, 1, dict_length))
            x1[0, 0, self.char_int_dict[char]] = 1
            x = np.append(x, x1, axis=1)
        # If plot is True print the predicted probabilities as well as the probability of getting 'E' in each step
        if plot:
            fig, ax = plt.subplots()
            for i in range(len(predictions)):
                # Plot probability distributions with opacity depending on the order in the sequence
                ax.plot(predictions[i, :], 'b', alpha=min(i * 0.01, 1))
            ax.set_xticks(np.arange(0, dict_length))
            ax.set_xticklabels(list(self.char_int_dict.keys()))
            plt.ylabel('Probability')
            plt.show()
            # The longer the sequence, the higher the probability to obtain 'E'
            plt.plot(predictions[:, -1], '-o')
            plt.ylabel('Probability')
            plt.xlabel('Sequence length')
            plt.show()
        return molecule_string
