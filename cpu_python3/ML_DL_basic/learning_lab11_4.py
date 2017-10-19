# CNN with MNIST dataset
'''
CNN 설계
1. Layer 1: CNN + Max pool
1-0. Input: MNIST image 28x28 size 이미지
1-1. CNN1
    - 3x3 Color 1개의 Filter 32개  ==> output: 32개
    - stride = 1x1 ==> output: 28x28 size
    - ReLU
1-2. Max pool1
    - 2x2 Filter, stride = 2x2 ==> output: 14x14 size x 32개

2. Layer 2: CNN + Max pool
2-0. Input: 14x14 size 32개 이미지
2-1. CNN2
    - 3x3 Color 1개의 Filter 64개 ==> output: 64개
    - stride = 1x1 ==> output: 14x14 size
    - ReLU
2-2. Max pool2
    - 2x2 Filter, stride = 2x2 ==> 7x7 size x 64개

3. Fully Connected Layer
3-0. Input: 7x7x64=3136
3-1. Logistic Classification 과 유사하게 결과값 연결
    - WX + b ==> Softmax ==> Cross-Entropy ==> Output: 10개

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

# 1-1. CNN
cnn1_input = tf.reshape(X, [-1, 28, 28, 1])             # 28x28x1
cnn1_W = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

cnn1_stride = [1, 1, 1, 1]
cnn1_padding = 'SAME'
cnn1_out = tf.nn.conv2d(cnn1_input, cnn1_W, strides=cnn1_stride, padding=cnn1_padding)  # 28x28x32
cnn1_out = tf.nn.relu(cnn1_out)

# 1-2. Max pool
pool1_ksize = [1, 2, 2, 1]
pool1_stride = [1, 2, 2, 1]
pool1_padding = 'SAME'
pool1_out = tf.nn.max_pool(cnn1_out, ksize=pool1_ksize, strides=pool1_ksize, padding=pool1_padding) # 14x14x32


# 2-1. CNN2
cnn2_W = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

cnn2_stride = [1, 1, 1, 1]
cnn2_padding = 'SAME'
cnn2_out = tf.nn.conv2d(pool1_out, cnn2_W, strides=cnn2_stride, padding=cnn2_padding)   # 14x14x64
cnn2_out = tf.nn.relu(cnn2_out)

# 2-2. Max pool
pool2_ksize = [1, 2, 2, 1]
pool2_stride = [1, 2, 2, 1]
pool2_padding = 'SAME'
pool2_out = tf.nn.max_pool(cnn2_out, ksize=pool2_ksize, strides=pool2_ksize, padding=pool2_padding) # 7x7x64

# 3-1. Logistic Classification 과 유사하게 결과값 연결
fc_input = tf.reshape(pool2_out, [-1, 7 * 7 * 64]) # 3136
fc_W = tf.get_variable("fc_W", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
fc_b = tf.Variable(tf.random_normal([10]))

fc_logits = tf.matmul(fc_input, fc_W) + fc_b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_logits, labels=Y))
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
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y})
        avg_cost += c / total_batch_count

    print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

print('Learning Finished!')


# Validation
correct_prediction = tf.equal(tf.argmax(fc_logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(fc_logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))


'''
Learning started. It takes sometime.
Epoch: 0001 cost= 0.377017410
Epoch: 0002 cost= 0.093983852
Epoch: 0003 cost= 0.068445598
Epoch: 0004 cost= 0.055149658
Epoch: 0005 cost= 0.046016893
Epoch: 0006 cost= 0.039835537
Epoch: 0007 cost= 0.035868216
Epoch: 0008 cost= 0.030284827
Epoch: 0009 cost= 0.027409474
Epoch: 0010 cost= 0.022958432
Epoch: 0011 cost= 0.021447448
Epoch: 0012 cost= 0.018324174
Epoch: 0013 cost= 0.016279432
Epoch: 0014 cost= 0.013826102
Epoch: 0015 cost= 0.012747095
Learning Finished!
Accuracy: 0.9859
Label:  [7]
Prediction:  [7]
'''
