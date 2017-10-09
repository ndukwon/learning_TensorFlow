# deep CNN with MNIST dataset
'''
deep CNN 설계
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
import random

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# 1-0. Input: MNIST image 28x28 size 이미지
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

X = tf.placeholder(tf.float32, [None, 784])             # X: m * 784 Matrix
Y = tf.placeholder(tf.float32, [None, 10])              # Y: m * 10 Matrix
keep_prob = tf.placeholder(tf.float32)

# 1-1. CNN1
layer1_cnn_input = tf.reshape(X, [-1, 28, 28, 1])             # 28x28x1
layer1_cnn_W = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

layer1_cnn_stride = [1, 1, 1, 1]
layer1_cnn_padding = 'SAME'
layer1_cnn_out = tf.nn.conv2d(layer1_cnn_input, layer1_cnn_W, strides=layer1_cnn_stride, padding=layer1_cnn_padding)  # 28x28x32
layer1_cnn_out = tf.nn.relu(layer1_cnn_out)

# 1-2. Max pool1
layer1_pool_ksize = [1, 2, 2, 1]
layer1_pool_stride = [1, 2, 2, 1]
layer1_pool_padding = 'SAME'
layer1_pool_out = tf.nn.max_pool(layer1_cnn_out, ksize=layer1_pool_ksize, strides=layer1_pool_stride, padding=layer1_pool_padding) # 14x14x32

# 1-3. dropout1
layer1_dropout = tf.nn.dropout(layer1_pool_out, keep_prob=keep_prob)


# 2-1. CNN2
layer2_cnn_W = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

layer2_cnn_stride = [1, 1, 1, 1]
layer2_cnn_padding = 'SAME'
layer2_cnn_out = tf.nn.conv2d(layer1_dropout, layer2_cnn_W, strides=layer2_cnn_stride, padding=layer2_cnn_padding)   # 14x14x64
layer2_cnn_out = tf.nn.relu(layer2_cnn_out)

# 2-2. Max pool2
layer2_pool_ksize = [1, 2, 2, 1]
layer2_pool_stride = [1, 2, 2, 1]
layer2_pool_padding = 'SAME'
layer2_pool_out = tf.nn.max_pool(layer2_cnn_out, ksize=layer2_pool_ksize, strides=layer2_pool_stride, padding=layer2_pool_padding) # 7x7x64

# 2-3. dropout2
layer2_dropout = tf.nn.dropout(layer2_pool_out, keep_prob=keep_prob)


# 3-1. CNN3
layer3_cnn_W = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))

layer3_cnn_stride = [1, 1, 1, 1]
layer3_cnn_padding = 'SAME'
layer3_cnn_out = tf.nn.conv2d(layer2_dropout, layer3_cnn_W, strides=layer3_cnn_stride, padding=layer3_cnn_padding)   # 7x7x128
layer3_cnn_out = tf.nn.relu(layer3_cnn_out)

# 3-2. Max pool3
layer3_pool_ksize = [1, 2, 2, 1]
layer3_pool_stride = [1, 2, 2, 1]
layer3_pool_padding = 'SAME'
layer3_pool_out = tf.nn.max_pool(layer3_cnn_out, ksize=layer3_pool_ksize, strides=layer3_pool_stride, padding=layer3_pool_padding) # 4x4x128

# 3-3. dropout3
layer3_dropout = tf.nn.dropout(layer3_pool_out, keep_prob=keep_prob)


# 4-1. Logistic Classification 과 유사하게 결과값 연결
layer4_fc_input = tf.reshape(layer3_dropout, [-1, 4 * 4 * 128]) # 2048
layer4_fc_W = tf.get_variable("layer4_fc_W", shape=[4 * 4 * 128, 625], initializer=tf.contrib.layers.xavier_initializer())
layer4_fc_b = tf.Variable(tf.random_normal([625]))

layer4_fc_out = tf.matmul(layer4_fc_input, layer4_fc_W) + layer4_fc_b
layer4_fc_out = tf.nn.relu(layer4_fc_out)

# 4-2. dropout4
layer4_dropout = tf.nn.dropout(layer4_fc_out, keep_prob=keep_prob)


# 5-1. Logistic Classification 과 유사하게 결과값 연결
layer5_fc_W = tf.get_variable("layer5_fc_W", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
layer5_fc_b = tf.Variable(tf.random_normal([10]))

layer5_fc_out = tf.matmul(layer4_dropout, layer5_fc_W) + layer5_fc_b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer5_fc_out, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch_count = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch_count):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.7})
        avg_cost += c / total_batch_count

    print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

print('Learning Finished!')


# Validation
correct_prediction = tf.equal(tf.argmax(layer5_fc_out, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(layer5_fc_out, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))


'''
Learning started. It takes sometime.
Epoch: 0001 cost= 0.291256698
Epoch: 0002 cost= 0.056332083
Epoch: 0003 cost= 0.036951848
Epoch: 0004 cost= 0.028124126
Epoch: 0005 cost= 0.023358903
Epoch: 0006 cost= 0.018616828
Epoch: 0007 cost= 0.015338304
Epoch: 0008 cost= 0.015232026
Epoch: 0009 cost= 0.011923661
Epoch: 0010 cost= 0.011210762
Epoch: 0011 cost= 0.008649805
Epoch: 0012 cost= 0.007334230
Epoch: 0013 cost= 0.006648044
Epoch: 0014 cost= 0.008441019
Epoch: 0015 cost= 0.005900943
Learning Finished!
Accuracy: 0.9906
Label:  [1]
Prediction:  [1]


# dropout 적용 후

Learning started. It takes sometime.
Epoch: 0001 cost= 0.370057208
Epoch: 0002 cost= 0.097895388
Epoch: 0003 cost= 0.072757328
Epoch: 0004 cost= 0.059838751
Epoch: 0005 cost= 0.053641720
Epoch: 0006 cost= 0.045906441
Epoch: 0007 cost= 0.043834122
Epoch: 0008 cost= 0.039287253
Epoch: 0009 cost= 0.034470695
Epoch: 0010 cost= 0.033388691
Epoch: 0011 cost= 0.031786095
Epoch: 0012 cost= 0.029760598
Epoch: 0013 cost= 0.028764733
Epoch: 0014 cost= 0.026594039
Epoch: 0015 cost= 0.027003375
Learning Finished!
Accuracy: 0.9929
Label:  [7]
Prediction:  [7]
'''
