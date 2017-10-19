# W에 따른 cost의 추이를 그래프로 표현하기

# Loading TensorFlow
import tensorflow as tf

import math

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Y =
# x_data = [-3, -2, -1, 1, 2, 3]
# y_data = [0, 0, 0, 1, 1, 1]
x_data = [-3, -2, -1, 1, 2, 3]
y_data = [1, 1, 1, 0, 0, 0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# W = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='weight')

# hypothesis = 1 / 1 + math.exp(-W * X)
# H(x) = 1 / (1 + e^(-W * X))
hh = X * W + b
hypothesis = tf.sigmoid(hh)

# cost = tf.reduce_mean(tf.square(hypothesis - Y))

# cost(W) = -1/m * ∑ (y * log(H(x)) + (1 - y) * log(1 - H(x))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))

# Session 생성
sess = tf.Session()

# Global variable 초기화
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})

    if step % 10 == 0:
        print(step, cost_val)

# 정확성 체크
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
print("Hypothesis:\n", h, "\nCorrect (Y):", c, "\nAccuracy:", a)

h_output = sess.run(hh, feed_dict={X: x_data})
hs_output = sess.run(hypothesis, feed_dict={X: x_data})

# matplotlib으로 표현하기
plt.scatter(x_data, y_data, c='black')
plt.xlabel('X')
plt.ylabel('Y')

# plt.plot(x_data, h_output, c='blue', marker='*')
# plt.plot(x_data, hs_output, c='red', marker='*')

plt.show()
