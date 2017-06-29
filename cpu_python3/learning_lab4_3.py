# Multi-Variable 학습시킬 값을 CSV 파일로 로딩

# Loading TensorFlow
import tensorflow as tf

# Loading Numpy
import numpy as np

# Loading CSV File
xy = np.loadtxt('data_01_test_score.csv', delimiter=',', dtype=np.float32)
'''
| EXAM1 | EXAM2 | EXAM3 | FINAL |
---------------------------------
|   73  |   80  |   75  |  152  |
|   93  |   88  |   93  |  185  |
|   89  |   91  |   90  |  180  |
|   96  |   98  |  100  |  196  |
|   73  |   66  |   70  |  142  |
---------------------------------
'''

# [열의 index나 범위, 행의 index나 범위]
x_data = xy[:, 0:-1]
y_data = xy[:,[-1]]

print(x_data.shape, x_data, len(x_data))
'''
(5, 3) [[  73.   80.   75.]
 [  93.   88.   93.]
 [  89.   91.   90.]
 [  96.   98.  100.]
 [  73.   66.   70.]] 5
'''
print(y_data.shape, y_data)
'''
(5, 1) [[ 152.]
 [ 185.]
 [ 180.]
 [ 196.]
 [ 142.]]
'''

# x1, x2, x3
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# w1, w2, w3
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)

print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print("Other scores will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
