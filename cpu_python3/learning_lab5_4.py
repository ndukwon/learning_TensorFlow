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
W = tf.placeholder(tf.float32)
# W = tf.Variable(tf.random_normal([2, 1]), name='weight')

# hypothesis = 1 / 1 + math.exp(-W * X)
# H(x) = 1 / (1 + e^(-W * X))
hypothesis = tf.sigmoid(X * W)

# cost = tf.reduce_mean(tf.square(hypothesis - Y))

# cost(W) = -1/m * ∑ (y * log(H(x)) + (1 - y) * log(1 - H(x))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Session 생성
sess = tf.Session()

# Global variable 초기화
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

# -3 ~ 5 까지 W(기울기)를 0.1씩 바꿔보면서 W에 따른 cost 값을 구한다
for i in range(-300, 300):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W, X:x_data, Y:y_data})
    print("curr_cost=", curr_cost)
    print("curr_W=", curr_W)

    if curr_cost != 'nan' and curr_cost != 'inf':
        W_val.append(curr_W)
        cost_val.append(curr_cost)

# matplotlib으로 표현하기
plt.plot(W_val, cost_val)
plt.show()
