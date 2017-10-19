# long sentence RNN training

'''
long sentence RNN 설계

1. Num of Input: ?, Num of Sequence(unfold): ?, Num of batchs: 1
   a. 문장에 포함된 모든 charactor로 중복없는 charactor index 사전을 만든다
   b. 문장을 사전의 Index로 치환한다.
   c. Index로 치환된 문장을 One-hot encoding으로 다시 치환한다.
2. Num of Output: 5(One-hot)
3. RNN
4. Cost
5. Training
6. Predict
'''

import tensorflow as tf
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()


# 1. Num of Input: ?, Num of Sequence(unfold): ?, Num of batchs: 1
sentence = (" if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

# a. 문장에 포함된 모든 charactor로 중복없는 charactor index 사전을 만든다
char_set = list(set(sentence))
char2idx_dic = {c:i for i, c in enumerate(char_set)}
print('char_set:', char_set)
print('char2idx_dic:', char2idx_dic)

input_dim = len(char2idx_dic)
hidden_size = len(char2idx_dic)
sequence_length = 15
learning_rate = 0.1

# b. 문장을 사전의 Index로 치환한다.
x_data = []
y_data = []
for i in range(len(sentence) - sequence_length) :
    sentence_window = sentence[i : i + sequence_length + 1]
    print('sentence_window:', sentence_window)
    data1 = sentence_window[:-1]
    data2 = sentence_window[1 :]
    print(i, data1, '->', data2)

    x_data_batch = [char2idx_dic[c] for c in data1]
    y_data_batch = [char2idx_dic[c] for c in data2]

    x_data.append(x_data_batch)
    y_data.append(y_data_batch)

x_data = np.reshape(x_data, (-1, sequence_length))
y_data = np.reshape(y_data, (-1, sequence_length))
print('x_data:', x_data)
batch_size = len(x_data)

X = tf.placeholder(tf.int32, [None, sequence_length])       # X label
Y = tf.placeholder(tf.int32, [None, sequence_length])       # Y label

# c. Index로 치환된 문장을 One-hot encoding으로 다시 치환한다.(TF가 이를 대체해줌)
x_one_hot = tf.one_hot(X, len(char2idx_dic))
print('x_one_hot:', x_one_hot)

# 3. RNN
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

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
    l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
    result = sess.run(prediction, feed_dict={X:x_data})

    if i%10 == 0:
        print(i, 'loss:', l, ' result:', result, ' True Y:', y_data)

        result_str = [char_set[index] for index in np.squeeze(result)]
        print('Prediction String:', ''.join(result_str))


'''
.. .. ..
sentence_window: mmensity of the
sentence_window: mensity of the 
sentence_window: ensity of the s
sentence_window: nsity of the se
sentence_window: sity of the sea
x_one_hot: Tensor("one_hot:0", shape=(?, 15, 25), dtype=float32)
25 LSTMStateTuple(c=25, h=25)
Traceback (most recent call last):
  File "/Users/dukwonnam/workspace/learning_TensorFlow/cpu_python3/learning_lab12_4.py", line 80, in <module>
    l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
  File "/Users/dukwonnam/tf_py_3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 895, in run
    run_metadata_ptr)
  File "/Users/dukwonnam/tf_py_3/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1100, in _run
    % (np_val.shape, subfeed_t.name, str(subfeed_t.get_shape())))
ValueError: Cannot feed value of shape (166, 14) for Tensor 'Placeholder:0', which has shape '(?, 15)'
'''
