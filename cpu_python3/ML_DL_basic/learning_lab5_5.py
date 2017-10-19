# Logistic classification

# Loading TensorFlow
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# shape 의미: X 는 2개 짜리 배열이 여러개(None) 들어간다
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

linear_hypothesis = tf.matmul(X, W) + b
# H(x) = 1 / (1 + e^(-W * X))
hypothesis = tf.sigmoid(linear_hypothesis)

# cost(W) = -1/m * ∑ (y * log(H(x)) + (1 - y) * log(1 - H(x))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})

        if step % 10 == 0:
            print(step, cost_val)

    # 정확성 체크
    w_vals = sess.run(W)
    b_val = sess.run(b)
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("Hypothesis:\n", h, "\nCorrect (Y):", c, "\nAccuracy:", a)

    last_linear_h, last_h = sess.run([linear_hypothesis, hypothesis], feed_dict={X:x_data, Y:y_data})
    print('last_linear_h', last_linear_h)
    print('last_h', last_h)

# x_numpy = np.array(x_data)
# plt.scatter(x_numpy[ : , [0]], x_numpy[ : , [1]])

x1_data_linear = []
x2_data_linear = []
y_data_linear = []
for x_points, y_point in zip(x_data, y_data):
    x1_data_linear.append(x_points[0])
    x2_data_linear.append(x_points[1])
    y_data_linear.append(y_point[0])

# indices = y_data_linear == 0
# colors = ['black','red']
# plt.scatter(x1_data_linear, x2_data_linear)
# plt.xlabel('X1')
# plt.ylabel('X2')
#
# # lineA = (1 - (w_vals[0] * x_points[0]) - b_val) / w_vals[1]
# # lineB = (-(w_vals[0] * x_points[0]) - b_val) / w_vals[1]
# # print('x1_point[', x_points[0], ']:', x_points[1], lineA, lineB)
# # plt.scatter(x_points[0], lineA, c='blue')
# # plt.scatter(x_points[0], lineB, c='green')
#
# plt.show()

x1_A = []
x1_B = []
x2_A = []
x2_B = []
lineA = []
lineB = []
for x1, x2, y_val in zip(x1_data_linear, x2_data_linear, y_data_linear):
    if y_val == 0 :
        x1_A.append(x1)
        x2_A.append(x2)
    else:
        x1_B.append(x1)
        x2_B.append(x2)

    lineA.append((-(w_vals[0] * x1) - b_val) / w_vals[1])
    lineB.append((1 - (w_vals[0] * x1) - b_val) / w_vals[1])

plt.xlabel('X1')
plt.ylabel('X2')
A_group = plt.scatter(x1_A, x2_A, c='black')
B_group = plt.scatter(x1_B, x2_B, c='red')
plt.scatter(x1_data_linear, lineA, c='blue')
plt.scatter(x1_data_linear, lineB, c='blue')
plt.legend((A_group, B_group),('Y==0','Y==1'))
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1_data_linear, x2_data_linear, y_data_linear, c='black')
# ax.scatter(x1_data_linear, x2_data_linear, last_linear_h, c='blue')
ax.scatter(x1_data_linear, x2_data_linear, last_h, c='red')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

plt.show()
