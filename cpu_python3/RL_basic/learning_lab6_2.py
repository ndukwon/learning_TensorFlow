# Cart Pole with random trial


import gym

env = gym.make('CartPole-v0')
env.reset()

num_episodes = 10

for i in range(num_episodes):
    done = False
    episode_reward = 0
    env.reset()

    while not done:
        env.render()

        action = env.action_space.sample()

        new_state, step_reward, done, info = env.step(action)
        print(new_state, step_reward, done)
        episode_reward += step_reward

        if done:
            print('Reward for', i,'episode was:', episode_reward)


'''
[-0.0486948   0.15871954 -0.00354478 -0.30057087] 1.0 False
[-0.04552041 -0.03635171 -0.00955619 -0.009008  ] 1.0 False
[-0.04624744 -0.23133532 -0.00973635  0.28064458] 1.0 False
[-0.05087415 -0.42631704 -0.00412346  0.57024086] 1.0 False
[-0.05940049 -0.2311375   0.00728135  0.27626175] 1.0 False
[-0.06402324 -0.42636258  0.01280659  0.57123229] 1.0 False
[-0.07255049 -0.23142253  0.02423124  0.28261124] 1.0 False
[-0.07717894 -0.42688158  0.02988346  0.58283702] 1.0 False
[-0.08571657 -0.23219076  0.0415402   0.29971561] 1.0 False
[-0.09036039 -0.42787944  0.04753451  0.60520472] 1.0 False
[-0.09891798 -0.62363276  0.05963861  0.91247291] 1.0 False
[-0.11139063 -0.81950866  0.07788807  1.22328776] 1.0 False
[-0.1277808  -1.01554269  0.10235382  1.53932326] 1.0 False
[-0.14809166 -1.21173624  0.13314029  1.86211291] 1.0 False
[-0.17232638 -1.01830157  0.17038254  1.61355648] 1.0 False
[-0.19269241 -1.21497561  0.20265367  1.95414622] 1.0 False
[-0.21699193 -1.41159236  0.2417366   2.30220918] 1.0 True
Reward for 9 episode was: 17.0
'''
