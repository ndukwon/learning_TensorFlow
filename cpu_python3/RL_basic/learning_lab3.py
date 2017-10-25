# Dummy Q-learning

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
num_episodes = 2000

episodeRewardList = []

for i in range(num_episodes):
    state = env.reset()
    episodeReward = 0
    done = False

    while not done:
        action = random_action(Q[state, :])
        new_state, stepReward, done, info = env.step(action)
        # env.render()    # Show the board after action

        Q[state, action] = stepReward + np.max(Q[new_state, :])

        episodeReward += stepReward
        state = new_state

    episodeRewardList.append(episodeReward)

print('Success rate:', str(sum(episodeRewardList)/num_episodes))
print('Final Q-table values')
print('LEFT DOWN RIGHT UP')
print(Q)
plt.bar(range(len(episodeRewardList)), episodeRewardList)
plt.show()


'''
Success rate: 0.9385
Final Q-table values
[[ 0.  1.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  1.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  1.  0.]
 [ 0.  0.  1.  0.]
 [ 0.  1.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  1.  0.]
 [ 0.  0.  0.  0.]]

'''
