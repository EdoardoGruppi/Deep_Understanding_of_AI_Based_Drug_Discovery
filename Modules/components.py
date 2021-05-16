# Import packages
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Sampling(Layer):
    def __init__(self, latent_rep_size, epsilon_std=0.01):
        """
        Custom layer aimed to sample the latent vectors from which to generate molecules.

        :param latent_rep_size: size of the latent vector produced by the encoder.
        :param epsilon_std: standard deviation of the normalized form which part of the latent vector is sampled.
        """
        super(Sampling, self).__init__()
        self.latent_rep_size = latent_rep_size
        self.epsilon_std = epsilon_std

    @tf.function
    def call(self, inputs):
        # Mean and log variance of the latent distribution
        z_mean_, z_log_var_ = inputs
        # Get the batch size
        batch_size = tf.shape(z_mean_)[0]
        # Sample latent vectors from the latent distribution
        epsilon = tf.random.normal(shape=(batch_size, self.latent_rep_size), mean=0, stddev=self.epsilon_std)
        return z_mean_ + tf.math.exp(z_log_var_ / 2) * epsilon
