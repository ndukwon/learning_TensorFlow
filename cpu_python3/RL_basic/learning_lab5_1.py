# OpenAI Gym, Frozen Lake
'''
'readchar' key input library is used
slippery O
'''
import readchar

# Macros
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT,
}



import gym
from gym.envs.registration import register

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    # kwargs={'map_name' : '4x4', 'is_slippery': False}
    kwargs={'map_name' : '4x4', 'is_slippery': True}
)

env = gym.make('FrozenLake-v3')
env.render()    # Show the initial board

while True:
    key = readchar.readkey()
    print('Now key:', key)

    if key not in arrow_keys.keys():
        print('Game aborted!')
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()    # Show the board after action
    print('State:', state, ', Reward:', reward, 'Done:', done, 'Info:', info)

    if done:
        print('Finished with reward:', reward)
        break
