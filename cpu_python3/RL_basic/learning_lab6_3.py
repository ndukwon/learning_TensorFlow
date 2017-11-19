# Cart Pole with Q Network

import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v0')
env.reset()

learning_rate = 0.1
discount = .99
num_episodes = 2000

# Model
input_size = env.observation_space.shape[0]       # input_size = 4
output_size = env.action_space.n                    # output_size = 2

X = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
# W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))      # 16 => 4
W1 = tf.get_variable('W1', shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())

Q_prediction = tf.matmul(X, W1)
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

loss = tf.square(Y - Q_prediction)
cost = tf.reduce_sum(loss)
optimizing = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

episode_reward_list = []
for i in range(num_episodes):
    done = False
    episode_reward = 0
    state = env.reset()
    step_count = 0

    e = 1/((i/100) + 1)

    while not done:
        step_count += 1
        # env.render()
        state_reshaped = np.reshape(state, [1, input_size])
        Q = sess.run(Q_prediction, feed_dict={X:state_reshaped})

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q)

        new_state, step_reward, done, info = env.step(action)
        # print(new_state, step_reward, done)

        if done:
            # Q[0, action] = step_reward
            Q[0, action] = -100     # 끝나길 원하지 않으므로
        else:
            new_state_reshaped = np.reshape(new_state, [1, input_size])
            next_Q = sess.run(Q_prediction, feed_dict={X:new_state_reshaped})
            Q[0, action] = step_reward + discount*np.max(next_Q)

        sess.run(optimizing, feed_dict={X:state_reshaped, Y:Q})

        episode_reward += step_reward
        state = new_state

    episode_reward_list.append(episode_reward)
    print('Episode: {} steps: {}'.format(i, step_count))
    if len(episode_reward_list) > 10 and np.mean(episode_reward_list[-10:]) > 500:
        break


done = False
episode_reward = 0
state = env.reset()
step_count = 0

while not done:
    step_count += 1
    # env.render()
    state_reshaped = np.reshape(state, [1, input_size])
    Q = sess.run(Q_prediction, feed_dict={X:state_reshaped})
    action = np.argmax(Q)

    new_state, step_reward, done, info = env.step(action)

    episode_reward += step_reward
    state = new_state

print('Total score: {}'.format(episode_reward))


'''
Episode: 0 steps: 14
Episode: 1 steps: 10
Episode: 2 steps: 14
Episode: 3 steps: 17
Episode: 4 steps: 21
Episode: 5 steps: 23
Episode: 6 steps: 16
Episode: 7 steps: 14
Episode: 8 steps: 22
Episode: 9 steps: 16
Total score: 152.0

Episode: 1995 steps: 9
Episode: 1996 steps: 9
Episode: 1997 steps: 47
Episode: 1998 steps: 18
Episode: 1999 steps: 15
Total score: 31.0
'''
