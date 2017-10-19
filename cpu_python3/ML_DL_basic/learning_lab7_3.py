# MNIST Dataset
'''
Multinormial Classification problem

'''

import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 0 ~ 9 에 대한 인식결과가 나와야 함
nb_classes = 10

# 28 * 28 pixel을 모두 X로 둔다
X = tf.placeholder(tf.float32, [None, 784])             # X: m * 784 Matrix
Y = tf.placeholder(tf.float32, [None, nb_classes])      # Y: m * 10 Matrix
W = tf.Variable(tf.random_normal([784, nb_classes]))    # W: 784 * 10 Matrix
b = tf.Variable(tf.random_normal([nb_classes]))         # b: 10-dimentional Vector

logits = tf.matmul(X, W) + b            # m * 10(for Y) Matrix
hypothesis = tf.nn.softmax(logits)      # m * 10(for Y) matrix

# Cross-Entropy = D(S, L) = - ∑ L.i * log(S.i)
cross_entropy = -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)  # SUM ((m * 10) .* (m * 10), 세로로) => 1 * 10 matrix
cost = tf.reduce_mean(cross_entropy)    # 1 Scalor

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


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
Epoch: 0001 cost= 2.762690731
Epoch: 0002 cost= 1.112676914
Epoch: 0003 cost= 0.893308130
Epoch: 0004 cost= 0.783082655
Epoch: 0005 cost= 0.712937049
Epoch: 0006 cost= 0.663862831
Epoch: 0007 cost= 0.625668863
Epoch: 0008 cost= 0.595375464
Epoch: 0009 cost= 0.570844348
Epoch: 0010 cost= 0.549369818
Epoch: 0011 cost= 0.531375442
Epoch: 0012 cost= 0.516004975
Epoch: 0013 cost= 0.501447339
Epoch: 0014 cost= 0.489145482
Epoch: 0015 cost= 0.478161628
Accuracy: 0.8891
label: [6]
Prediction: [6]
'''
