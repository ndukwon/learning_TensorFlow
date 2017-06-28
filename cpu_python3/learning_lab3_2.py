# GradientDescentOptimizer를 직접 해보기

# Loading TensorFlow
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name="weight")
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# H(x) = Wx
hypothesis = X * W

# Cost: 1/m * ∑(H(x) - y)^2
# cost = tf.reduce_sum(tf.square(hypothesis - Y))  동영상 강의내용 중 오타로 보임
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent 구현
# 한번 학습을 반영하는 비율
learning_rate = 0.1

# Cost의 미분(cost의 기울기)
gradient = tf.reduce_mean((W * X - Y) * X)

# 학습된 W(linear 함수의 기울기)를 반영
descent = W - learning_rate * gradient
update = W.assign(descent)

# Session 생성
sess = tf.Session()

# Global variable 초기화
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


'''
0 3.39134 [ 0.14752436]
1 0.964646 [ 0.54534638]
2 0.274388 [ 0.75751805]
3 0.0780483 [ 0.87067628]
4 0.0222004 [ 0.93102735]
5 0.00631479 [ 0.96321458]
6 0.0017962 [ 0.98038113]
7 0.00051092 [ 0.98953658]
8 0.000145329 [ 0.99441952]
9 4.13377e-05 [ 0.99702376]
10 1.1758e-05 [ 0.99841267]
11 3.34437e-06 [ 0.99915344]
12 9.51278e-07 [ 0.99954849]
13 2.70601e-07 [ 0.9997592]
14 7.69795e-08 [ 0.99987155]
15 2.18962e-08 [ 0.99993151]
16 6.23436e-09 [ 0.99996346]
17 1.77049e-09 [ 0.99998051]
18 4.99488e-10 [ 0.99998963]
19 1.44057e-10 [ 0.99999446]
20 4.21636e-11 [ 0.99999702]
'''
