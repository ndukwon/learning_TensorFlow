# Q-learning
'''
e-greedy

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

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')
# env.render()    # Show the initial board

Q = np.zeros([env.observation_space.n, env.action_space.n])
discount = .99
num_episodes = 2000

episode_reward_list = []

for i in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    e = 1/((i//100) + 1)

    while not done:
        # 어쩔때는 최소값대로 어쩔때는 모험을 하도록
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else :
            action = np.argmax(Q[state, :])
        new_state, step_reward, done, info = env.step(action)
        # env.render()    # Show the board after action

        Q[state, action] = step_reward + discount * np.max(Q[new_state, :])

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
Success rate: 0.8195
Final Q-table values
LEFT DOWN RIGHT UP
[[ 0.94148015  0.95099005  0.95099005  0.94148015]
 [ 0.94148015  0.          0.96059601  0.95099005]
 [ 0.95099005  0.970299    0.          0.96059601]
 [ 0.96059601  0.          0.          0.        ]
 [ 0.95099005  0.96059601  0.          0.94148015]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.9801      0.          0.96059601]
 [ 0.          0.          0.          0.        ]
 [ 0.96059601  0.          0.970299    0.95099005]
 [ 0.96059601  0.9801      0.9801      0.        ]
 [ 0.970299    0.99        0.          0.970299  ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.9801      0.99        0.970299  ]
 [ 0.9801      0.99        1.          0.9801    ]
 [ 0.          0.          0.          0.        ]]

Process finished with exit code 0

'''
