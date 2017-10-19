# ML lab 09-1: Neural Net for XOR
# with Neural network
# + Deep NN

import tensorflow as tf
import numpy as np

'''
X1 | X2 |  Y
---|----|----
 0 |  0 |  0
 1 |  0 |  1
 0 |  1 |  1
 1 |  1 |  0
'''
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
y_data = np.array([   [0],    [1],    [1],    [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')

# hypothesis
a2 = tf.sigmoid(tf.matmul(X, W1) + b1)
a3 = tf.sigmoid(tf.matmul(a2, W2) + b2)
a4 = tf.sigmoid(tf.matmul(a3, W3) + b3)
hypothesis = a4

# cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Prediction / Accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            current_cost, current_W1, current_W2, current_W3, current_W4 = sess.run([cost, W1, W2, W3, W4], feed_dict={X: x_data, Y: y_data})
            print(step, current_cost, current_W1, current_W2, current_W3, current_W4)

    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("hypothesis:", h, "\npredicted:", p, "\naccuracy:", a)


'''
hypothesis: [[ 0.01784295  0.02014822  0.02220584  0.01834023  0.02193335  0.02348019
   0.02514559  0.02297441  0.01807683  0.01262918]
 [ 0.97688282  0.9781999   0.9758755   0.97920132  0.97130203  0.96981961
   0.97033018  0.96756649  0.97637707  0.97783649]
 [ 0.97958791  0.97495788  0.97411126  0.98212147  0.96977735  0.97413504
   0.98271805  0.97824395  0.98116702  0.98256242]
 [ 0.02798621  0.02741132  0.0284477   0.0215213   0.03815178  0.0350179
   0.02281922  0.03191997  0.02520591  0.02951136]]
predicted: [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
accuracy: 1.0
'''
