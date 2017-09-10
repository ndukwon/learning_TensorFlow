# ML lab 09-1: Neural Net for XOR
# with Logistic regression

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
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

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
            current_cost, current_W = sess.run([cost, W], feed_dict={X: x_data, Y: y_data})
            print(step, current_cost, current_W)

    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("hypothesis:", h, "\npredicted:", p, "\naccuracy:", a)

'''
hypothesis: [[ 0.5]
 [ 0.5]
 [ 0.5]
 [ 0.5]]
predicted: [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
accuracy: 0.5
'''
