# deep CNN with MNIST dataset
'''
# tensorflow.layers 로 deep CNN in Model class 설계
# 그 후 Model 여러개를 학습시키고 그 값을 조합하는 Ensemble을 수행
1. Layer 1: CNN + Max pool
1-0. Input: MNIST image 28x28 size 이미지
1-1. CNN1
    - 3x3 Color 1개의 Filter 32개  ==> output: 32개
    - stride = 1x1 ==> output: 28x28 size
    - ReLU
1-2. Max pool1
    - 2x2 Filter, stride = 2x2 ==> output: 14x14 size x 32개
1-3. dropout1

2. Layer 2: CNN + Max pool
2-0. Input: 14x14 size 32개 이미지
2-1. CNN2
    - 3x3 Color 1개의 Filter 64개 ==> output: 64개
    - stride = 1x1 ==> output: 14x14 size
    - ReLU
2-2. Max pool2
    - 2x2 Filter, stride = 2x2 ==> output: 7x7 size x 64개
2-3. dropout2

3. Layer 3: CNN + Max pool
3-0. Input: 7x7 size 64개 이미지
3-1. CNN3
    - 3x3 Color 1개의 Filter 128개 ==> output: 128개
    - stride = 1x1 ==> output: 7x7 size
    - ReLU
3-2. Max pool3
    - 2x2 Filter, stride = 2x2 ==> output: 4x4 size x 128개
3-3. dropout3

4. Fully Connected Layer 1
4-0. Input: 4x4x128=2048
4-1. Logistic Classification 과 유사하게 결과값 연결
    - WX + b ==> Output: 625개
    - ReLU
4-2. dropout4

5. Fully Connected Layer 2
5-0. Input: 625
5-1. Logistic Classification 과 유사하게 결과값 연결
    - WX + b ==> Output: 10개
    - Softmax
    - Cross-Entropy

'''

import tensorflow as tf
import numpy as np

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# 1-0. Input: MNIST image 28x28 size 이미지
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 784])             # X: m * 784 Matrix
            self.Y = tf.placeholder(tf.float32, [None, 10])              # Y: m * 10 Matrix
            self.keep_prob = tf.placeholder(tf.float32)
            self.training = tf.placeholder(tf.bool)

            # 1-1. CNN1
            layer1_cnn_input = tf.reshape(self.X, [-1, 28, 28, 1])             # 28x28x1

            layer1_cnn_filter = 32
            layer1_cnn_kernel_size = [3, 3]
            layer1_cnn_padding = 'SAME'
            layer1_cnn_out = tf.layers.conv2d(inputs=layer1_cnn_input,
                                              filters=layer1_cnn_filter,
                                              kernel_size=layer1_cnn_kernel_size,
                                              padding=layer1_cnn_padding,
                                              activation=tf.nn.relu)
            # 1-2. Max pool1
            layer1_pool_size = [2, 2]
            layer1_pool_stride = 2
            layer1_pool_padding = 'SAME'
            layer1_pool_out = tf.layers.max_pooling2d(inputs=layer1_cnn_out,
                                                      pool_size=layer1_pool_size,
                                                      padding=layer1_pool_padding,
                                                      strides=layer1_pool_stride)
            # 1-3. dropout1
            layer1_dropout = tf.layers.dropout(inputs=layer1_pool_out,
                                               rate=self.keep_prob,
                                               training=self.training)


            # 2-1. CNN2
            layer2_cnn_filter = 64
            layer2_cnn_kernel_size = [3, 3]
            layer2_cnn_padding = 'SAME'
            layer2_cnn_out = tf.layers.conv2d(inputs=layer1_dropout,
                                              filters=layer2_cnn_filter,
                                              kernel_size=layer2_cnn_kernel_size,
                                              padding=layer2_cnn_padding,
                                              activation=tf.nn.relu)
            # 2-2. Max pool2
            layer2_pool_size = [2, 2]
            layer2_pool_stride = 2
            layer2_pool_padding = 'SAME'
            layer2_pool_out = tf.layers.max_pooling2d(inputs=layer2_cnn_out,
                                                      pool_size=layer2_pool_size,
                                                      padding=layer2_pool_padding,
                                                      strides=layer2_pool_stride)
            # 2-3. dropout2
            layer2_dropout = tf.layers.dropout(inputs=layer2_pool_out,
                                               rate=self.keep_prob,
                                               training=self.training)


            # 3-1. CNN3
            layer3_cnn_filter = 128
            layer3_cnn_kernel_size = [3, 3]
            layer3_cnn_padding = 'SAME'
            layer3_cnn_out = tf.layers.conv2d(inputs=layer2_dropout,
                                              filters=layer3_cnn_filter,
                                              kernel_size=layer3_cnn_kernel_size,
                                              padding=layer3_cnn_padding,
                                              activation=tf.nn.relu)
            # 3-2. Max pool3
            layer3_pool_size = [2, 2]
            layer3_pool_stride = 2
            layer3_pool_padding = 'SAME'
            layer3_pool_out = tf.layers.max_pooling2d(inputs=layer3_cnn_out,
                                                      pool_size=layer3_pool_size,
                                                      padding=layer3_pool_padding,
                                                      strides=layer3_pool_stride)
            # 3-3. dropout3
            layer3_dropout = tf.layers.dropout(inputs=layer3_pool_out,
                                               rate=self.keep_prob,
                                               training=self.training)


            # 4-1. Logistic Classification 과 유사하게 결과값 연결
            layer4_fc_input = tf.reshape(layer3_dropout, [-1, 4 * 4 * 128]) # 2048

            layer4_fc_unit = 625
            layer4_fc_out = tf.layers.dense(inputs=layer4_fc_input,
                                            units=layer4_fc_unit,
                                            activation=tf.nn.relu)

            # 4-2. dropout4
            layer4_dropout = tf.layers.dropout(inputs=layer4_fc_out,
                                               rate=self.keep_prob,
                                               training=self.training)


            # 5-1. Logistic Classification 과 유사하게 결과값 연결
            layer5_fc_unit = 10
            self.logits = tf.layers.dense(inputs=layer4_dropout,
                                            units=layer5_fc_unit)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Validation
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prob=1, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test,
                                                                   self.keep_prob: keep_prob,
                                                                   self.training: training})

    def cal_accuracy(self, x_test, y_test, keep_prob=1, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test,
                                                       self.Y: y_test,
                                                       self.keep_prob: keep_prob,
                                                       self.training: training})

    def train(self, x_train, y_train, keep_prob=0.7, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_train,
                                                                     self.Y: y_train,
                                                                     self.keep_prob: keep_prob,
                                                                     self.training: training})


# Training
sess = tf.Session()
models = []
num_of_models = 2
for m_index in range(num_of_models) :
    models.append(Model(sess, "m" + str(m_index)))
sess.run(tf.global_variables_initializer())

print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch_count = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch_count):

        for m_index, m in enumerate(models) :
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            c, _ = m.train(batch_x, batch_y)
            avg_cost_list[m_index] += c / total_batch_count

    print('Epoch:', '%04d' % (epoch + 1), 'cost=', avg_cost_list)

print('Learning Finished!')


# Validation
test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10])
for m_index, m in enumerate(models) :
    print(m_index, 'Accuracy:', m.cal_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images, mnist.test.labels)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# for m_index, m in enumerate(models) :
#     p = m.predict(mnist.test.images[r:r + 1])
#     print('Prediction[' + m_index + ']: ', )


'''
Learning started. It takes sometime.
Epoch: 0001 cost= [ 0.9015705   0.90658544]
Epoch: 0002 cost= [ 0.32954114  0.35397362]
Epoch: 0003 cost= [ 0.26364763  0.27113406]
Epoch: 0004 cost= [ 0.22797151  0.24150017]
Epoch: 0005 cost= [ 0.209574    0.21881782]
Epoch: 0006 cost= [ 0.19816198  0.19627066]
Epoch: 0007 cost= [ 0.18928568  0.19232015]
Epoch: 0008 cost= [ 0.18517801  0.18467542]
Epoch: 0009 cost= [ 0.17817999  0.18314625]
Epoch: 0010 cost= [ 0.16806504  0.18058072]
Epoch: 0011 cost= [ 0.16860375  0.17050981]
Epoch: 0012 cost= [ 0.16587909  0.16618038]
Epoch: 0013 cost= [ 0.16238155  0.16724899]
Epoch: 0014 cost= [ 0.16285091  0.16010729]
Epoch: 0015 cost= [ 0.16022233  0.15846316]
Learning Finished!
0 Accuracy: 0.9877
1 Accuracy: 0.9885
Ensemble accuracy: 0.9889
'''
