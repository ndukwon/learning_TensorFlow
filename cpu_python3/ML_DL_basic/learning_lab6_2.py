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

print("x_data=", x_data)
print("y_data=", y_data)

y_min = min(y_data)
y_max = max(y_data)
# Y 가 0 ~ 6 까지의 범위
nb_classes = y_max - y_min + 1
print(y_max, y_min, nb_classes)

_, x_shape = x_data.shape
_, y_shape = y_data.shape
print(x_shape, y_shape)

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

# def getXYplaceholder(x_shape, y_shape):
#     X = tf.placeholder(tf.float32, [None, x_shape])
#     Y = tf.placeholder(tf.int32, [None, y_shape])
#     return X, Y
#
# X, Y = getXYplaceholder(x_shape, y_shape)

print("X=", X)
print("Y=", Y)
y_one_hot = tf.one_hot(Y, nb_classes)
print("one hot", y_one_hot)
y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes])
print("reshape", y_one_hot)

def getWbVariable(x_shape, nb_classes):
    W = tf.Variable(tf.random_normal([x_shape, nb_classes]), name='weight')
    b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
    return W, b

W, b = getWbVariable(x_shape, nb_classes)

# 1-2) H(X) = WX + b (= logits)
logits = tf.matmul(X, W) + b

# 1-3) Softmax(H(X)) = S(y.i) = e^(y.i) / (∑ e^y)
hypothesis = tf.nn.softmax(logits)

# 2-3) Cross-Entropy = D(S, L) = - ∑ L.i * log(S.i)
def getCost1(hypothesis):
    cross_entropy = -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)
    cost = tf.reduce_mean(cross_entropy)
    return cost

def getCost2(logits, y_one_hot):
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
    cost = tf.reduce_mean(cost_i)
    return cost

# cost = getCost1(hypothesis)
cost = getCost2(logits, y_one_hot)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 1-4) One-hot encoding
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction), tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})

            print("Step: {:5}\tLoss: {:3f}\tAcc: {:.2%}".format(step, loss, acc))


    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
