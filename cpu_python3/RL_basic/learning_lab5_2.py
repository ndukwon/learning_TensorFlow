# Q-learning
'''
Stochastic

'''

import numpy as np
import random as pr
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt

def random_action(vector):
    '''
    Choose randomly the index of the list(random argmax)
    1. max 값을 찾아서
    2. ??

    :param vector: a list that (i.e.) contains LEFT, TOP, RIGHT, BOTTOM
    :return: a int the randomly selected index of vector
    '''

    m = np.amax(vector)
    indice = np.nonzero(vector == m)[0]
    return pr.choice(indice)

# register(
#     id='FrozenLake-v3',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': True}
# )

env = gym.make('FrozenLake-v0')
# env.render()    # Show the initial board

Q = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = 0.1
discount = .99
num_episodes = 2000

episode_reward_list = []

for i in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    e = 1/((i/100) + 1)

    while not done:
        # 어쩔때는 최소값대로 어쩔때는 모험을 하도록
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else :
            action = np.argmax(Q[state, :])
        new_state, step_reward, done, info = env.step(action)
        # env.render()    # Show the board after action

        # Q[state, action] = step_reward + discount * np.max(Q[new_state, :])
        Q[state, action] = (1 - learning_rate)*Q[state, action] + learning_rate*(step_reward + discount*np.max(Q[new_state, :]))

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

Before

Success rate: 0.024
Final Q-table values
LEFT DOWN RIGHT UP
[[  2.84557859e-10   0.00000000e+00   0.00000000e+00   2.81712281e-10]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   2.99222751e-10]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   2.81712281e-10]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   2.84557859e-10]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  9.41480149e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  9.90000000e-01   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00   0.00000000e+00]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]]


After

Success rate: 0.3175
Final Q-table values
LEFT DOWN RIGHT UP
[[ 0.57500622  0.52581947  0.51875611  0.5363787 ]
 [ 0.21726843  0.19531523  0.14202888  0.42367125]
 [ 0.34347734  0.17927396  0.16502914  0.12141723]
 [ 0.04115953  0.02277452  0.0042193   0.01209362]
 [ 0.59162304  0.35235086  0.4921031   0.36952076]
 [ 0.          0.          0.          0.        ]
 [ 0.34666247  0.08032307  0.14741377  0.05809853]
 [ 0.          0.          0.          0.        ]
 [ 0.50877404  0.32647258  0.47101188  0.60900403]
 [ 0.49690759  0.64931842  0.50464785  0.43968206]
 [ 0.67175213  0.45342201  0.39950975  0.15602843]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.46881868  0.45948667  0.75915494  0.55067804]
 [ 0.66576072  0.88327384  0.83310524  0.74353318]
 [ 0.          0.          0.          0.        ]]

Process finished with exit code 0


Process finished with exit code 0


'''
