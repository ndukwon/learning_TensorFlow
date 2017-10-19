# 'if you want you' RNN training

'''
'if you want you' RNN 설계

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
sess = tf.InteractiveSession()


# 1. Num of Input: ?, Num of Sequence(unfold): ?, Num of batchs: 1
sample = ' if you want you'

# a. 문장에 포함된 모든 charactor로 중복없는 charactor index 사전을 만든다
char_set = list(set(sample))
char2idx_dic = {c:i for i, c in enumerate(char_set)}
print('char_set:', char_set)
print('char2idx_dic:', char2idx_dic)

# b. 문장을 사전의 Index로 치환한다.
sample2idx = [char2idx_dic[c] for c in sample]
print('sample2idx:', sample2idx)
x_data = [sample2idx[:-1]]
y_data = [sample2idx[1:]]

input_dim = len(char2idx_dic)
hidden_size = len(char2idx_dic)
batch_size = 1                      # one sentence
sequence_length = len(sample) - 1   # 마지막 캐릭터는 예측할 필요가 없으므로
learning_rate = 0.1

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
char_set: ['o', 'a', 'i', 'w', 'y', 'u', 'n', ' ', 'f', 't']
char2idx_dic: {'o': 0, 'a': 1, 'i': 2, 'w': 3, 'y': 4, 'u': 5, 'n': 6, ' ': 7, 'f': 8, 't': 9}
sample2idx: [7, 2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]

x_one_hot: Tensor("one_hot:0", shape=(?, 15, 10), dtype=float32)
10 LSTMStateTuple(c=10, h=10)
0 loss: 2.30404  result: [[7 5 5 5 5 5 5 5 5 5 5 5 0 5 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String:  uuuuuuuuuuuouu
10 loss: 1.43938  result: [[2 8 7 4 0 5 7 3 1 6 9 7 4 0 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String: if you want you
20 loss: 1.17186  result: [[2 8 7 4 0 5 7 3 1 6 9 7 4 0 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String: if you want you
30 loss: 1.10102  result: [[2 8 7 4 0 5 7 3 1 6 9 7 4 0 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String: if you want you
40 loss: 1.08074  result: [[2 8 7 4 0 5 7 3 1 6 9 7 4 0 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String: if you want you
50 loss: 1.07166  result: [[2 8 7 4 0 5 7 3 1 6 9 7 4 0 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String: if you want you
60 loss: 1.0657  result: [[2 8 7 4 0 5 7 3 1 6 9 7 4 0 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String: if you want you
70 loss: 1.05983  result: [[2 8 7 4 0 5 7 3 1 6 9 7 4 0 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String: if you want you
80 loss: 1.05594  result: [[2 8 7 4 0 5 7 3 1 6 9 7 4 0 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String: if you want you
90 loss: 1.05387  result: [[2 8 7 4 0 5 7 3 1 6 9 7 4 0 5]]  True Y: [[2, 8, 7, 4, 0, 5, 7, 3, 1, 6, 9, 7, 4, 0, 5]]
Prediction String: if you want you
'''
