# W에 따른 cost의 추이를 그래프로 표현하기

# TensorFlow 실행
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# H(x) = Wx
hypothesis = X * W

# Cost: 1/m * ∑(H(x) - y)^2
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Session 생성
sess = tf.Session()

# Global variable 초기화
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

# -3 ~ 5 까지 W(기울기)를 0.1씩 바꿔보면서 W에 따른 cost 값을 구한다
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# matplotlib으로 표현하기
plt.plot(W_val, cost_val)
plt.show()
