# 'hihello' RNN training

'''
'hihello' RNN 설계

1. Num of Input: 5(One-hot), Num of Sequence(unfold): 6, Num of batchs: 1
   - (몇종류의 X(단어)를 학습시킬 것인지, X 1개의 Sequence(RNN이 Unfold 되는, 옆으로 늘어나는) 개수, feature의 개수)
2. Num of Output: 5(One-hot)
3. RNN
4. Cost
5. Training
6. Predict
'''

import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()


num_classes = 5
input_dim = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
learning_rate = 0.1

# 1. Num of Input: 5(One-hot), Num of Sequence(unfold): 6, Num of batchs: 1
idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]       # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3
y_data = [[1, 0, 2, 3, 3, 4]]       # ihello

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])               # Y label


# 3. RNN
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

print(cell.output_size, cell.state_size)

# 4. Cost
weight = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weight)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

# 5. Training
sess.run(tf.global_variables_initializer())

for i in range(100):
    l, _ = sess.run([loss, train], feed_dict={X:x_one_hot, Y:y_data})
    result = sess.run(prediction, feed_dict={X:x_one_hot})

    if i%10 == 0:
        print(i, 'loss:', l, ' result:', result, ' True Y:', y_data)

        result_str = [idx2char[index] for index in np.squeeze(result)]
        print('Prediction String:', ''.join(result_str))


'''
5 LSTMStateTuple(c=5, h=5)
0 loss: 1.64429  result: [[2 3 2 2 4 4]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: eleeoo
10 loss: 1.09188  result: [[1 3 3 3 3 3]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: illlll
20 loss: 0.881115  result: [[1 3 2 3 3 3]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: ilelll
30 loss: 0.817307  result: [[1 0 2 3 3 3]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: ihelll
40 loss: 0.736221  result: [[1 0 2 3 3 4]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: ihello
50 loss: 0.688797  result: [[1 0 2 3 3 4]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: ihello
60 loss: 0.675263  result: [[1 0 2 3 3 4]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: ihello
70 loss: 0.666125  result: [[1 0 2 3 3 4]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: ihello
80 loss: 0.659442  result: [[1 0 2 3 3 4]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: ihello
90 loss: 0.656512  result: [[1 0 2 3 3 4]]  True Y: [[1, 0, 2, 3, 3, 4]]
Prediction String: ihello
'''
