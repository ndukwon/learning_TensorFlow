# GradientDescentOptimizer를 직접 해보기

# TensorFlow 실행
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

# 한번 학습을 반영하는 비율
learning_rate = 0.1

# Cost의 기울기
gradient = tf.reduce_mean((W * X - Y) * X)

# 학습된 기울기를 반영
descent = W - learning_rate * gradient
update = W.assign(descent)

# Session 생성
sess = tf.Session()

# Global variable 초기화
sess.run(tf.global_variables_initializer())

for step in range(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))


'''
0 4.37343 [ 0.03192812]
1 1.244 [ 0.48369494]
2 0.353848 [ 0.72463727]
3 0.10065 [ 0.85313988]
4 0.0286294 [ 0.92167461]
5 0.00814347 [ 0.95822644]
6 0.00231636 [ 0.9777208]
7 0.000658875 [ 0.98811775]
8 0.000187416 [ 0.99366277]
9 5.33082e-05 [ 0.99662018]
10 1.51635e-05 [ 0.99819744]
11 4.31314e-06 [ 0.99903864]
12 1.2269e-06 [ 0.99948728]
13 3.48991e-07 [ 0.99972653]
14 9.92565e-08 [ 0.99985415]
15 2.82443e-08 [ 0.99992222]
16 8.03129e-09 [ 0.99995852]
17 2.27936e-09 [ 0.99997789]
18 6.47167e-10 [ 0.9999882]
19 1.87796e-10 [ 0.99999368]
'''
