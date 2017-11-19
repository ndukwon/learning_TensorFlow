# Frozen Lake with Q Network
'''

'''

import numpy as np
import gym
# from gym.envs.registration import register
import tensorflow as tf
import matplotlib.pyplot as plt

# register(
#     id='FrozenLake-v3',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': True}
# )

env = gym.make('FrozenLake-v0')
# env.render()    # Show the initial board

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1
discount = .99
num_episodes = 2000

def one_hot(x):
    return np.identity(input_size)[x:x+1]

X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))      # 16 => 4

Q_prediction = tf.matmul(X, W)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

loss = tf.square(Y - Q_prediction)
cost = tf.reduce_sum(loss)
gradient = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

episode_reward_list = []
for i in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    e = 1/((i/100) + 1)

    while not done:
        Q = sess.run(Q_prediction, feed_dict={X:one_hot(state)})

        # 어쩔때는 최소값대로 어쩔때는 모험을 하도록
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else :
            action = np.argmax(Q)
        new_state, step_reward, done, info = env.step(action)
        # env.render()    # Show the board after action

        if done:
            Q[0, action] = step_reward
        else:
            next_Q = sess.run(Q_prediction, feed_dict={X:one_hot(new_state)})
            Q[0, action] = step_reward + discount*np.max(next_Q)

        sess.run(gradient, feed_dict={X:one_hot(state), Y: Q})

        episode_reward += step_reward
        state = new_state

    episode_reward_list.append(episode_reward)

print('Success rate:', str(sum(episode_reward_list)/num_episodes))
print('Final Q-table values')
print('LEFT DOWN RIGHT UP')
print(Q)
plt.bar(range(len(episode_reward_list)), episode_reward_list)
plt.show()


'''
Success rate: 0.321
Final Q-table values
LEFT DOWN RIGHT UP
[[ 0.          0.08994664  0.20177756  0.16833116]]
'''
