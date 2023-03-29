import tensorflow as tf
import numpy as np
from models.loss_functions import LevinLoss, CrossEntropyLoss,\
    CrossEntropyMSELoss, LevinMSELoss, MSELoss, ImprovedLevinLoss,\
    ImprovedLevinMSELoss, RegLevinLoss, RegLevinMSELoss

class InvalidLossFunction(Exception):
    pass

"""
def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))
"""

class HeuristicFeedForwardNet(tf.keras.Model):

    def __init__(self, layer_size, number_actions, reg_const=0.001):
        tf.keras.backend.set_floatx('float64')

        super(HeuristicFeedForwardNet, self).__init__(name='')

        self._reg_const = reg_const
        self._number_actions = number_actions

        self.ff1 = tf.keras.layers.Dense(layer_size,
                                            name='ff1',
                                            activation='relu',
                                            dtype='float64')
        self.ff2 = tf.keras.layers.Dense(layer_size,
                                            name='ff2',
                                            activation='relu',
                                            dtype='float64')
        self.ff3 = tf.keras.layers.Dense(layer_size,
                                            name='ff3',
                                            activation='relu',
                                            dtype='float64')


        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(128,
                                            name='dense1',
                                            activation='relu',
                                            dtype='float64')
        self.dense2 = tf.keras.layers.Dense(1,
                                            name='dense2',
                                            dtype='float64')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self._loss_function = MSELoss()

    def predict(self, x):
        return self.call(x).numpy()

    def call(self, input_tensor):

        x = self.ff1(input_tensor)
        x = self.ff2(x)
        x = self.ff3(x)
        x_dropout = self.dropout(x, training=True)
        x = self.dense1(x_dropout)
        logits_h = self.dense2(x)

        return logits_h

    def train_with_memory(self, memory):
        losses = []
        memory.shuffle_trajectories()
        for trajectory in memory.next_trajectory():

            with tf.GradientTape() as tape:
                loss = self._loss_function.compute_loss(trajectory, self)

            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            losses.append(loss)

        return np.mean(losses)

    def get_number_actions(self):
        return self._number_actions

class TwoHeadedFeedForwardNet(tf.keras.Model):

    def __init__(self, layer_size, number_actions, loss_name, reg_const=0.001):
        tf.keras.backend.set_floatx('float64')

        super(TwoHeadedFeedForwardNet, self).__init__(name='')

        self._reg_const = reg_const
        self._number_actions = number_actions
        self._loss_name = loss_name

        self.ff1 = tf.keras.layers.Dense(layer_size,
                                            name='ff1',
                                            activation='relu',
                                            dtype='float64')
        self.ff2 = tf.keras.layers.Dense(layer_size,
                                            name='ff2',
                                            activation='relu',
                                            dtype='float64')
        self.ff3 = tf.keras.layers.Dense(layer_size,
                                            name='ff3',
                                            activation='relu',
                                            dtype='float64')


        self.dropout = tf.keras.layers.Dropout(0.5)

        #Probability distribution
        self.dense11 = tf.keras.layers.Dense(128,
                                             name='dense11',
                                             activation='relu',
                                             dtype='float64')
        self.dense12 = tf.keras.layers.Dense(number_actions,
                                             name='dense12',
                                             dtype='float64')

        #Heuristic value
        self.dense21 = tf.keras.layers.Dense(128,
                                             name='dense21',
                                             activation='relu',
                                             dtype='float64')
        self.dense22 = tf.keras.layers.Dense(1,
                                             name='dense22',
                                             dtype='float64')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self._loss_name = loss_name

        if loss_name == 'LevinLoss':
            self._loss_function = LevinMSELoss()
        elif loss_name == 'CrossEntropyLoss':
            self._loss_function = CrossEntropyMSELoss()
        elif loss_name == 'ImprovedLevinLoss':
            self._loss_function = ImprovedLevinMSELoss()
        elif loss_name == 'RegLevinLoss':
            self._loss_function = RegLevinMSELoss()
        else:
            raise InvalidLossFunction

    def predict(self, x):
        log_softmax, x_softmax, _, pred_h = self.call(x)
        return log_softmax.numpy(), x_softmax.numpy(), pred_h.numpy()

    def multiple_predict(self, x, num_samples = 10):
        x = np.repeat(x, num_samples, axis = 0)
        return self.call(x, training = True)

    def call(self, input_tensor, training = False):

        x = self.ff1(input_tensor)
        x = self.ff2(x)
        x = self.ff3(x)

        x_dropout = self.dropout(x, training = training)
        x1 = self.dense11(x_dropout)
        logits_pi = self.dense12(x1)
        x_log_softmax = tf.nn.log_softmax(logits_pi)
        x_softmax = tf.nn.softmax(logits_pi)

        x2 = self.dense21(x_dropout)
        logits_h = self.dense22(x2)

        return x_log_softmax, x_softmax, logits_pi, logits_h

    def train_with_memory(self, memory):
        losses = []
        memory.shuffle_trajectories()

        for trajectory in memory.next_trajectory():

            with tf.GradientTape() as tape:
                loss = self._loss_function.compute_loss(trajectory, self)

            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            losses.append(loss)

        return np.mean(losses)

    def get_number_actions(self):
        return self._number_actions

class FeedForwardNet(tf.keras.Model):

    def __init__(self, layer_size, number_actions, loss_name, reg_const=0.001):
        tf.keras.backend.set_floatx('float64')

        super(FeedForwardNet, self).__init__(name='')

        self._max_grad_norms = []

        self._reg_const = reg_const
        self._number_actions = number_actions
        self._loss_name = loss_name

        self.ff1 = tf.keras.layers.Dense(layer_size,
                                            name='ff1',
                                            activation='relu',
                                            dtype='float64')
        self.ff2 = tf.keras.layers.Dense(layer_size,
                                            name='ff2',
                                            activation='relu',
                                            dtype='float64')
        self.ff3 = tf.keras.layers.Dense(layer_size,
                                            name='ff3',
                                            activation='relu',
                                            dtype='float64')

        self.dense1 = tf.keras.layers.Dense(128,
                                            name='dense1',
                                            activation='relu',
                                            dtype='float64')
        self.dense2 = tf.keras.layers.Dense(number_actions,
                                            name='dense2',
                                            dtype='float64')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        if loss_name == 'LevinLoss':
            self._loss_function = LevinLoss()
        elif loss_name == 'ImprovedLevinLoss':
            self._loss_function = ImprovedLevinLoss()
        elif loss_name == 'CrossEntropyLoss':
            self._loss_function = CrossEntropyLoss()
        elif loss_name == 'RegLevinLoss':
            self._loss_function = RegLevinLoss()
        else:
            raise InvalidLossFunction

    def predict(self, x):
        log_softmax, x_softmax, _ = self.call(x)
        return log_softmax.numpy(), x_softmax.numpy()

    def call(self, input_tensor):

        x = self.ff1(input_tensor)
        x = self.ff2(x)
        x = self.ff3(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        x_softmax = tf.nn.softmax(logits)
        x_log_softmax = tf.nn.log_softmax(logits)

        return x_log_softmax, x_softmax, logits

    def _cross_entropy_loss(self, states, y):
        images = [s.get_image_representation() for s in states]
        _, _, logits = self(np.array(images))
        return self.cross_entropy_loss(y, logits)

    def train_with_memory(self, memory):
        losses = []
        memory.shuffle_trajectories()
        for trajectory in memory.next_trajectory():

            with tf.GradientTape() as tape:
                loss = self._loss_function.compute_loss(trajectory, self)

            grads = tape.gradient(loss, self.trainable_weights)

            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            losses.append(loss)

        return np.mean(losses)

    def train(self, states, y):
        with tf.GradientTape() as tape:
            loss = self._cross_entropy_loss(states, y)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    def get_number_actions(self):
        return self._number_actions
