
# Loading TensorFlow
import tensorflow as tf

# xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
# Filename Queue로 연결하기
filename_queue = tf.train.string_input_producer(['data-03-diabetes.csv'], shuffle=False, name='filename_queue')

# Filename Queue를 Reader에 연결하기
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Reader에 Decoder 연결하기
recode_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=recode_defaults)

# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]
# Decoder에 batch 단위 설정 연결하기
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# shape 의미: X 는 8개 짜리 배열이 여러개(None) 들어간다
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# H(x) = 1 / (1 + e^(-W * X))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost(W) = -1/m * ∑ (y * log(H(x)) + (1 - y) * log(1 - H(x))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_batch, Y:y_batch})

        if step % 10 == 0:
            print(step, cost_val)

    # 정확성 체크
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_batch, Y:y_batch})
    print("Hypothesis:\n", h, "\nCorrect (Y):", c, "\nAccuracy:", a)

    coord.request_stop()
    coord.join()
