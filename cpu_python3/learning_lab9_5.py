# ML lab 09-5: TensorBoard

import tensorflow as tf
import numpy as np

'''
X1 | X2 |  Y
---|----|----
 0 |  0 |  0
 1 |  0 |  1
 0 |  1 |  1
 1 |  1 |  0
'''
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
y_data = np.array([   [0],    [1],    [1],    [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')

    # hypothesis of layer 1
    a2 = tf.sigmoid(tf.matmul(X, W1) + b1)

    # logging layer 1 for TensorBoard
    W1_hist = tf.summary.histogram("W1_hist", W1)
    b1_hist = tf.summary.histogram("b1_hist", b1)
    a2_hist = tf.summary.histogram("a2_hist", a2)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')

    # the final hypothesis
    hypothesis = tf.sigmoid(tf.matmul(a2, W2) + b2)

    # logging layer 2 for TensorBoard
    W2_hist = tf.summary.histogram("W2_hist", W2)
    b2_hist = tf.summary.histogram("b2_hist", b2)
    hypothesis_hist = tf.summary.histogram("hyphthesis_hist", hypothesis)

with tf.name_scope("cost") as scope:
    # cost
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

    # logging cost for TensorBoard
    cost_sum = tf.summary.scalar("cost", cost)

with tf.name_scope("train") as scope:
    # train
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Prediction / Accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# logging cost for TensorBoard
accuracy_sum = tf.summary.scalar("accuracy", cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/lab9_5")
    writer.add_graph(sess.graph)  # Show the graph

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            current_cost, current_W1, current_W2 = sess.run([cost, W1, W2], feed_dict={X: x_data, Y: y_data})
            print(step, current_cost, current_W1, current_W2)

    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("hypothesis:", h, "\npredicted:", p, "\naccuracy:", a)


'''
Tensorboard는 그래프로 볼 수 있게 하는 툴
임의로 로그를 찍어야 하며 찍은 기록에 대해서 웹페이지 분석도구를 통해 보여주는 것

virtualenv를 사용할 때

1. virtualenv에서 python 2.7 버전을 실행(물론 여기에 TensorFlow가 설치되어 있는 상태)
(python 3 에서는 에러남)
$ source ~/tf_py_2.7/bin/activate

2. tensorboard를 실행
$ tensorboard --logdir=[로그가 있는 디렉토리 경로]

3. 웹페이지(로컬 또는 리모트) 열어서 확인
- 로컬: 127.0.0.1:6006
'''
