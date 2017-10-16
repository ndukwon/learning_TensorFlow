# basic RNN trial

import tensorflow as tf
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

with tf.variable_scope('one_cell') as scope:
    '''
    단순히 basic RNN을 한번 만들어서 학습하지 않고 돌려본다
    1. Num of Input: 4
    2. Num of Output: 2
    3. basic RNN
    4. Run RNN
    '''

    # 1. Num of Input: 4
    h = [1, 0, 0, 0]
    e = [0, 1, 0, 0]
    l = [0, 0, 1, 0]
    o = [0, 0, 0, 1]
    # x_data = 1x1x4 (몇종류의 X 문자 Sequence를 학습시킬 것인지, RNN이 Unfold 되는(옆으로 늘어나는) 개수, feature의 개수)
    x_data = np.array([[h]], dtype=np.float32)
    print(x_data.shape)
    pp.pprint(x_data)


    # 2. Num of Output: 2
    hidden_size = 2


    # 3. basic RNN
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    print(cell.output_size, cell.state_size)


    # 4. Run RNN
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())


    '''
    array([[[ 1.,  0.,  0.,  0.]]], dtype=float32)
    2 LSTMStateTuple(c=2, h=2)
    array([[[ 0.0405211 , -0.02904973]]], dtype=float32)
    '''

with tf.variable_scope('two_sequances') as scope:
    '''
    단순히 basic RNN을 한번 만들어서 학습하지 않고 돌려본다
    1. Num of Input: 4, Num of Sequence(unfold): 5
    2. Num of Output: 2
    3. basic RNN
    4. Run RNN
    '''
    # Num of Input: 4, Num of Sequence(unfold): 5
    h = [1, 0, 0, 0]
    e = [0, 1, 0, 0]
    l = [0, 0, 1, 0]
    o = [0, 0, 0, 1]
    # x_data = 1x5x4 (몇종류의 X 문자 Sequence를 학습시킬 것인지, RNN이 Unfold 되는(옆으로 늘어나는) 개수, feature의 개수)
    x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
    print(x_data.shape)
    pp.pprint(x_data)


    # 2. Num of Output: 2
    hidden_size = 2

    # 3. basic RNN
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    print(cell.output_size, cell.state_size)


    # 4. Run RNN
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

    '''
    (1, 5, 4)
    array([[[ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  1.]]], dtype=float32)
    2 LSTMStateTuple(c=2, h=2)
    array([[[-0.1067895 ,  0.00094427],
            [-0.17050879,  0.00149258],
            [-0.28006673, -0.10670394],
            [-0.34575421, -0.17221697],
            [-0.5020979 , -0.07536053]]], dtype=float32)
    '''

with tf.variable_scope('3_batches') as scope:
    '''
    단순히 basic RNN을 한번 만들어서 학습하지 않고 돌려본다
    1. Num of Input: 4, Num of Sequence(unfold): 5, Num of batchs: 3
    2. Num of Output: 2
    3. basic RNN
    4. Run RNN
    '''
    # Num of Input: 4, Num of Sequence(unfold): 5, Num of batchs: 3
    h = [1, 0, 0, 0]
    e = [0, 1, 0, 0]
    l = [0, 0, 1, 0]
    o = [0, 0, 0, 1]
    # x_data = 1x5x4 (몇종류의 X 문자 Sequence를 학습시킬 것인지, RNN이 Unfold 되는(옆으로 늘어나는) 개수, feature의 개수)
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    print(x_data.shape)
    pp.pprint(x_data)


    # 2. Num of Output: 2
    hidden_size = 2

    # 3. basic RNN
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    print(cell.output_size, cell.state_size)


    # 4. Run RNN
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

    '''
    (3, 5, 4)
    array([[[ 1.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  1.]],
    
           [[ 0.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  1.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.]],
    
           [[ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  1.,  0.,  0.],
            [ 0.,  0.,  1.,  0.]]], dtype=float32)
    2 LSTMStateTuple(c=2, h=2)
    array([[[ 0.05285873,  0.03601465],
            [ 0.09199385,  0.09054453],
            [ 0.1193665 , -0.04651929],
            [ 0.13100763, -0.15573123],
            [ 0.24914317, -0.25065768]],
    
           [[ 0.05584953,  0.06969149],
            [ 0.17982183, -0.11261501],
            [ 0.14927609, -0.18390596],
            [ 0.16019158, -0.23867126],
            [ 0.17062382, -0.2727837 ]],
    
           [[ 0.03388982, -0.13274252],
            [ 0.0656547 , -0.21352182],
            [ 0.10546419, -0.01597384],
            [ 0.14792851,  0.07157635],
            [ 0.17623261, -0.06058116]]], dtype=float32)
    '''

with tf.variable_scope('sequence_loss') as scope:
    y_data = tf.constant([[1, 1, 1]])     # 결과값 종류의 Index

    # index가 0이냐 1이냐를 예측
    predict1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]], dtype=tf.float32)
    predict2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)

    weight = tf.constant([[1, 1, 1]], dtype=tf.float32)

    sequence_loss1 = tf.contrib.seq2seq.sequence_loss(predict1, y_data, weight)
    sequence_loss2 = tf.contrib.seq2seq.sequence_loss(predict2, y_data, weight)

    sess.run(tf.global_variables_initializer())
    print('Loss1:', sequence_loss1.eval())
    print('Loss2:', sequence_loss2.eval())

    '''
    Loss1: 0.513015
    Loss2: 0.371101
    '''
