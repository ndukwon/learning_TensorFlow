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
0 [-37.333332, -3.0, [(-37.333336, -3.0)]]
1 [-2.4888866, 0.73333359, [(-2.4888866, 0.73333359)]]
2 [-0.1659257, 0.98222226, [(-0.16592571, 0.98222226)]]
3 [-0.011061668, 0.99881482, [(-0.011061668, 0.99881482)]]
4 [-0.00073742867, 0.99992096, [(-0.00073742867, 0.99992096)]]
5 [-4.9630802e-05, 0.9999947, [(-4.9630802e-05, 0.9999947)]]
6 [-3.0994415e-06, 0.99999964, [(-3.0994415e-06, 0.99999964)]]
7 [-6.7551929e-07, 0.99999994, [(-6.7551935e-07, 0.99999994)]]
8 [0.0, 1.0, [(0.0, 1.0)]]
9 [0.0, 1.0, [(0.0, 1.0)]]
'''
