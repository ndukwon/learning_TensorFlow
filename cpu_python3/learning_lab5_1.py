

# Loading TensorFlow
import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# shape 의미: X 는 2개 짜리 배열이 여러개(None) 들어간다
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# H(x) = 1 / (1 + e^(-W * X))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost(W) = -1/m * ∑ (y * log(H(x)) + (1 - y) * log(1 - H(x))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast()
