# Import packages
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from spektral.layers import ECCConv, GlobalSumPool
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
import tensorflow as tf


class FirstModel:
    def __init__(self, node_features, edge_features):
        """
        Create the model.

        :param node_features: number of features linked to every node.
        :param edge_features: number of features linked to every node.
        """
        # X is the node feature matrix represented by a np.array of shape (n_nodes, n_node_features)
        x_in = Input(shape=(node_features,), name='X_in')
        # A is the adjacency matrix, usually a scipy.sparse matrix of shape (n_nodes, n_nodes)
        a_in = Input(shape=(None,), sparse=True, name='A_in')
        # E is the the edge features represented in a sparse edge list format as a np.array of shape (n_edges,
        # n_edge_features)
        e_in = Input(shape=(edge_features,), name='E_in')
        # Vector to keep track of the different graphs in the disjoint union, for convolutional layers it is not needed.
        i_in = Input(shape=(), name='segment_ids_in', dtype=tf.int32)
        # Define model
        x_1 = ECCConv(32, activation='relu')([x_in, a_in, e_in])
        x_2 = ECCConv(32, activation='relu')([x_1, a_in, e_in])
        x_3 = GlobalSumPool()([x_2, i_in])
        outputs = Dense(units=1)(x_3)
        # Build model
        self.model = Model(inputs=[x_in, a_in, e_in, i_in], outputs=outputs)
        self.loss_fn = MeanSquaredError()

    def train(self, loader_tr, learning_rate):

        # Set the learning rate and the optimizer
        opt = Adam(lr=learning_rate)

        @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
        def train_step(inputs, target):
            with tf.GradientTape() as tape:
                # Return the results related to some inputs. Training=True allows some layers such as dropout or
                # batch normalization to work properly.
                predictions = self.model(inputs, training=True)
                # Compute the loss for the batch
                loss = self.loss_fn(target, predictions)
                loss += sum(self.model.losses)
            # use gradient to retrieve the gradients of the trainable variables with respect to the loss
            gradients = tape.gradient(loss, self.model.trainable_variables)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss
            opt.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss

        print("Fitting model...")
        current_batch, model_loss = 0, 0
        # Iterate over the batches
        for batch in loader_tr:
            # Compute the loss related to the current batch
            outs = train_step(*batch)
            # Update the loss and the batch count
            model_loss += outs
            current_batch += 1
            if not current_batch % loader_tr.steps_per_epoch:
                print(f"Epoch: {int(current_batch / loader_tr.steps_per_epoch)} - " 
                      f"Loss: {model_loss / loader_tr.steps_per_epoch:.3f}")
                # Restart the loss parameters after every epoch
                model_loss = 0

    def test(self, loader_te):
        print("\nTesting model...")
        model_loss = 0
        # For every batch to evaluate...
        for batch in loader_te:
            inputs, target = batch
            # Get the predictions and compute the loss
            predictions = self.model(inputs, training=False)
            model_loss += self.loss_fn(target, predictions)
        # Return the average batch loss
        model_loss /= loader_te.steps_per_epoch
        print(f"Test loss: {model_loss}")
