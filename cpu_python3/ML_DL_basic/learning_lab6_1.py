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

# 1-1) X, Y
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# 1-2) H(X) = WX + b (= logits)
logits = tf.matmul(X, W) + b

# 1-3) Softmax(H(X)) = S(y.i) = e^(y.i) / (∑ e^y)
hypothesis = tf.nn.softmax(logits)

# 2-3) Cross-Entropy = D(S, L) = - ∑ L.i * log(S.i)
cross_entropy = -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})

        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    print('-----------------------')
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    # 1-4) One-hot encoding
    print(a, sess.run(tf.arg_max(a, 1)))

    print('-----------------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.arg_max(b, 1)))

    print('-----------------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.arg_max(c, 1)))

    print('-----------------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.arg_max(all, 1)))



'''
0 5.35014
200 0.519889
400 0.413495
600 0.33367
800 0.259555
1000 0.228144
1200 0.206798
1400 0.189099
1600 0.174153
1800 0.161356
2000 0.150272
-----------------------
[[  2.98787490e-03   9.97001827e-01   1.02830218e-05]] [1]
-----------------------
[[ 0.88816816  0.09890945  0.01292237]] [0]
-----------------------
[[  9.80914994e-09   3.36761412e-04   9.99663234e-01]] [2]
-----------------------
[[  2.98787490e-03   9.97001827e-01   1.02830218e-05]
 [  8.88168156e-01   9.89094451e-02   1.29223689e-02]
 [  9.80914994e-09   3.36761412e-04   9.99663234e-01]] [1 0 2]
'''
