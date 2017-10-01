# MNIST Dataset
'''
Multinormial Classification problem
With neural network and ReLU
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

# tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 0 ~ 9 에 대한 인식결과가 나와야 함
nb_classes = 10
hidden_classes = 256

# 28 * 28 pixel을 모두 X로 둔다
X = tf.placeholder(tf.float32, [None, 784])             # X: m * 784 Matrix
Y = tf.placeholder(tf.float32, [None, nb_classes])      # Y: m * 10 Matrix

# Layer 1
W1 = tf.Variable(tf.random_normal([784, hidden_classes]))    # W: 784 * 256 Matrix
b1 = tf.Variable(tf.random_normal([hidden_classes]))         # b: 256-dimentional Vector

# Layer 2
z2 = tf.matmul(X, W1) + b1
a2 = tf.nn.relu(z2)
W2 = tf.Variable(tf.random_normal([hidden_classes, hidden_classes]))    # W: 784 * 256 Matrix
b2 = tf.Variable(tf.random_normal([hidden_classes]))                    # b: 256-dimentional Vector

# Layer 3
z3 = tf.matmul(a2, W2) + b2
a3 = tf.nn.relu(z3)
W3 = tf.Variable(tf.random_normal([hidden_classes, nb_classes]))    # W: 784 * 256 Matrix
b3 = tf.Variable(tf.random_normal([nb_classes]))                    # b: 256-dimentional Vector

# Layer 4
a4 = tf.matmul(a3, W3) + b3
hypothesis = a4     # m * 10(for Y) matrix

# Cross-Entropy = D(S, L) = - ∑ L.i * log(S.i)
# cross_entropy = -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)  # SUM ((m * 10) .* (m * 10), 세로로) => 1 * 10 matrix
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)
cost = tf.reduce_mean(cross_entropy)    # 1 Scalor

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


# Test hypothesis
is_currect = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))  # m * 1
accuracy = tf.reduce_mean(tf.cast(is_currect, tf.float32))

# Parameter
# epoch: 전체 데이터 m개를 모두 1회 학습시키는 것에 대한 단위, 1 epoch = 전체 한번 학습
training_epochs = 15

# batch size: 전체 데이터 m개를 분할하여 학습시키는 단위
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch_count = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch_count):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            cost_val, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / total_batch_count

        print('Epoch:', '%04d'%(epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

    # accuracy.eval() == sess.run(accuracy)
    print('Accuracy:', accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print('label:', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print('Prediction:', sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()


'''
Epoch: 0001 cost= 142.426676695
Epoch: 0002 cost= 38.811085416
Epoch: 0003 cost= 24.322322562
Epoch: 0004 cost= 16.859828000
Epoch: 0005 cost= 12.303971603
Epoch: 0006 cost= 9.098134073
Epoch: 0007 cost= 6.894147496
Epoch: 0008 cost= 5.085947123
Epoch: 0009 cost= 3.897444666
Epoch: 0010 cost= 2.903342553
Epoch: 0011 cost= 2.137495041
Epoch: 0012 cost= 1.653781394
Epoch: 0013 cost= 1.229289456
Epoch: 0014 cost= 1.030873799
Epoch: 0015 cost= 0.797019765
Accuracy: 0.9438
label: [8]
Prediction: [8]
'''
