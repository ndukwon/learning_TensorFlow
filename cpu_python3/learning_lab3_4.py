# gradient를 계산한 값 구해서 적용하기(compute_gradient, apply_gradient)

# Loading TensorFlow
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W = tf.Variable(5.0)
W = tf.Variable(-3.0)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# H(x) = Wx
hypothesis = X * W

# Cost: 1/m * ∑(H(x) - y)^2
cost = tf.reduce_mean(tf.square(hypothesis - Y))
gradient = tf.reduce_mean(((W * X - Y) * X) * 2)

# Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# Gradient 계산값 구하여 수동 적용
# train = optimizer.minimize(cost)
gvs = optimizer.compute_gradients(cost)
# TODO: gvs 수정
train = optimizer.apply_gradients(gvs)

# Session 생성
sess = tf.Session()

# Global variable 초기화
sess.run(tf.global_variables_initializer())

for step in range(10):
    print(step, sess.run([gradient, W, gvs], feed_dict={X: x_data, Y: y_data}))
    sess.run(train, feed_dict={X: x_data, Y: y_data})

'''
Result

W = tf.Variable(5.0)
0 5.0
1 1.26667
2 1.01778
3 1.00119
4 1.00008
5 1.00001
6 1.0
7 1.0
8 1.0
9 1.0

W = tf.Variable(-3.0)
0 -3.0
1 0.733334
2 0.982222
3 0.998815
4 0.999921
5 0.999995
6 1.0
7 1.0
8 1.0
9 1.0
'''
