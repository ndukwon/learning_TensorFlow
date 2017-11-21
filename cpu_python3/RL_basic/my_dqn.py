import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, sess, input_size, output_size, name='main'):
        self.sess = sess
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, hidden_size=10, learning_rate=0.1):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
            self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            W1 = tf.get_variable('W1', shape=[self.input_size, hidden_size], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            Z1 = tf.matmul(self._X, W1)
            A1 = tf.nn.tanh(Z1)

            W2 = tf.get_variable('W2', shape=[hidden_size, self.output_size], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

            self.Q_predict = tf.matmul(A1, W2)

        loss = tf.square(self._Y - self.Q_predict)
        self._cost = tf.reduce_mean(loss)
        self._optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._cost)

    def transform_to_X_data(self, states):
        x_data = np.reshape(states, [-1, self.input_size])
        return x_data

    def transform_to_Y_data(self, Q_reals):
        y_data = np.reshape(Q_reals, [-1, self.output_size])
        return y_data

    def predict(self, state):
        x_data = self.transform_to_X_data(state)
        return self.sess.run(self.Q_predict, feed_dict={self._X: x_data})

    def update(self, state_stack, Q_real_stack):
        x_data = state_stack
        y_data = Q_real_stack
        return self.sess.run([self._cost, self._optimize], feed_dict={self._X: x_data, self._Y:y_data})
