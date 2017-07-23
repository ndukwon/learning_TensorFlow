# Softmax

'''
1. 예측모델 구성 단계
1-1) X, Y
1-2) H(X) = WX + b (= logits)
1-3) Softmax(H(X)) = S(y.i) = e^(y.i) / (∑ e^y)
1-4) One-hot encoding

2. Cost 함수 구성
2-1) Softmax(H(X)) (= 예측모델 3단계)
2-2) Y값
2-3) Cross-Entropy = D(S, L) = - ∑ L.i * log(S.i)
'''

import tensorflow as tf
import numpy as np

def getXYdata():
    # Predicting animal type based on various features
    xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
    return xy[:, 0:-1], xy[:, [-1]]

x_data, y_data = getXYdata()

y_min = min(y_data)
y_max = max(y_data)
# Y 가 0 ~ 6 까지의 범위
nb_classes = y_max - y_min + 1
print(y_max, y_min, nb_classes)

_, x_shape = x_data.shape
_, y_shape = y_data.shape
print(x_shape, y_shape)

def getXYplaceholder(x_shape, y_shape):
    X = tf.placeholder(tf.float32, [None, x_shape])
    Y = tf.placeholder(tf.int32, [None, y_shape])
    return X, Y

X, Y = getXYplaceholder(x_shape, y_shape)

y_one_hot = tf.one_hot(Y, nb_classes)
y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes])

def getWbVariable(x_shape, nb_classes):
    W = tf.Variable(tf.random_normal([x_shape, nb_classes]), name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

W, b = getWbVariable(x_shape, nb_classes)

# 1-2) H(X) = WX + b (= logits)
logits = tf.matmul(X, W) + b

# 1-3) Softmax(H(X)) = S(y.i) = e^(y.i) / (∑ e^y)
hypothesis = tf.nn.softmax(logits)

# 2-3) Cross-Entropy = D(S, L) = - ∑ L.i * log(S.i)
# cross_entropy = -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
cost = tf.reduce_mean(cost_i)
