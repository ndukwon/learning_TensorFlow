# TensorFlow tips
'''
1. Overshooting: learning rate 너무크게 설정되어 Gradient Descent Algorithm을 적용해도 최적을 찾지 못하고 거꾸로 올라가는 현상
2. Small learning rate: 너무 조금씩 이동해서 시간이 오래걸리거나, 최적을 찾기도 전에 멈추게 된다
3. Normalization: 분포나 상대적 비율을 고르게
4. Overfitting: 너무 Training data에만 딱맞는 예측모델을 만드는 경우.
  - More training data
  - Feature의 개수를 줄인다
  - Regularization(일반화: 구부리는 예측모델보다는 직선을 사용하자.

Training datasets
'''

import tensorflow as tf

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

X = tf.placeholder(tf.float32, [None, 3])   # m * 3 matrix
Y = tf.placeholder(tf.float32, [None, 3])   # m * 3 matrix
W = tf.Variable(tf.random_normal([3, 3]))   # 3(for X) * 3(for Y) matrix
b = tf.Variable(tf.random_normal([3]))      # 3 Vector

# Softmax(H(X)) = S(y.i) = e^(y.i) / (∑ e^y)
logits = tf.matmul(X, W) + b                # m * 3(for Y) matrix
hypothesis = tf.nn.softmax(logits)          # m * 3(for Y) matrix

# Cross-Entropy = D(S, L) = - ∑ L.i * log(S.i)
cross_entropy = -tf.reduce_sum(Y * tf.log(hypothesis), axis=1)  # SUM ((m * 3) .* (m * 3), 세로로) => 1 * 3 matrix
cost = tf.reduce_mean(cross_entropy)        # 1 Scalor

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.5).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y:y_data})
        print(step, cost_val, W_val)

    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    print("Accuracy:", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))


'''
# learning rate = 0.1
...
198 0.608165 [[-1.18621242 -0.60341996  1.78477859]
 [ 0.77818787  0.58631778  0.57969272]
 [ 0.8318212   0.67340893 -0.16127108]]
199 0.607367 [[-1.19044018 -0.6036433   1.78922963]
 [ 0.77855414  0.58676124  0.57888305]
 [ 0.83312523  0.67324734 -0.16241352]]
200 0.606573 [[-1.19465792 -0.60385996  1.79366398]
 [ 0.7789157   0.58719951  0.57808322]
 [ 0.83443081  0.6730867  -0.16355848]]
Prediction: [2 2 2]
Accuracy: 1.0


# learning rate = 1.5 => Overshooting
...
3 14.3904 [[-0.42629606  0.75412488 -1.09141707]
 [ 1.63827515  1.35397816 -3.77843046]
 [ 2.96296549  1.94282794 -3.99989963]]
4 13.65 [[-1.45502019  1.22034907 -0.52891707]
 [-2.26141858  3.75367212 -2.27843046]
 [-1.0186522   4.61194611 -2.68739963]]
5 nan [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]
6 nan [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]
...
Prediction: [0 0 0]
Accuracy: 0.0


# learning rate = 0.001 => Not converged
...
198 3.21582 [[ 1.58356023 -0.05937053  0.75516963]
 [ 0.45609063 -0.65800029  1.33848   ]
 [ 0.33568716  0.04995758 -0.47435734]]
199 3.2056 [[ 1.58326352 -0.05900619  0.75510204]
 [ 0.45552582 -0.65627432  1.3373189 ]
 [ 0.33482867  0.05180301 -0.4753443 ]]
200 3.19539 [[ 1.58296633 -0.05864199  0.75503504]
 [ 0.45495814 -0.6545487   1.33616102]
 [ 0.33396769  0.05364792 -0.47632822]]
Prediction: [0 0 0]
Accuracy: 0.0

'''
