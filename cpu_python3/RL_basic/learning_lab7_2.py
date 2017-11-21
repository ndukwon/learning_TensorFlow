# Cart Pole with DQN 2015 year ver
'''
1. build model
    - X: 주어진 input 개수
    - Y: 주어진 output 개수
    - W: hidden 10개
    - predict: 주어진 output 개수
    - loss, cost, optimize 구현

2. Play episodes
    - E-greedy: 랜덤 탐험과 기존의 학습을 바탕으로 다음 Action을 선정
    - Play
    - 3. Play 된 정보를 저장
    - episode가 끝나면 4. 저장된 Play된 정보를 랜덤으로 불러와서 학습
    - 이러한 episode를 여러번 수행

3. Play 된 정보를 저장
    - Queue에 저장
    - data가 지나치게 많아질 수 있고 앞부분을 버린다.

4. 저장된 Play된 정보를 랜덤으로 불러와서 학습
    - Play된 정보를 Queue에서 랜덤으로 10개를 꺼내옴
    - 10개의 Predict Q(Y hat)을 기록된 state 정보로 구함
    - 10개의 Y를 기록된 next_state 정보로 구함
    - 이렇게 구한 Predict Q와 Y를 이용하여 optimize

'''

import numpy as np
import tensorflow as tf
from collections import deque
import my_dqn
import random

import gym
env = gym.make('CartPole-v0')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

DISCOUNT = 0.9
RECODING_MAX = 50000

def episode_play(env, episode, DQN, replay_buffer):
    state = env.reset()
    step_count=0
    done=False
    e = 1./((episode/10) + 1)

    while not done:
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(DQN.predict(state))

        next_state, reward, done, info = env.step(action)

        if done:
            reward = -100

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > RECODING_MAX:
            replay_buffer.popleft()

        state = next_state
        step_count += 1
        if step_count > 10000:      # 이 정도면 오래 살렸다.
            break
    print('Episode {} steps {}'.format(episode, step_count))
    return step_count


def episode_optimize(DQN, targetDQN, replay_buffer):
    for i in range(50):
        train_mini_batch = random.sample(replay_buffer, 10)

        state_stack = np.empty(0).reshape(0, DQN.input_size)
        Q_real_stack = np.empty(0).reshape(0, DQN.output_size)
        for state, action, reward, next_state, done in train_mini_batch:
            Q = DQN.predict(state)

            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + DISCOUNT*np.max(targetDQN.predict(next_state))

            state_stack = np.vstack([state_stack, state])
            Q_real_stack = np.vstack([Q_real_stack, Q])

        cost, _ = DQN.update(state_stack, Q_real_stack)
        print('Loss:', cost)

def get_copy_var_ops(dest_scope_name='target', src_scope_name='main'):
    ops = []

    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        ops.append(dest_var.assign(src_var.value()))

    return ops


def main():
    max_episode = 2500
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = my_dqn.DQN(sess, input_size, output_size, name='main')
        targetDQN = my_dqn.DQN(sess, input_size, output_size, name='target')
        sess.run(tf.global_variables_initializer())

        copy_ops = get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
        sess.run(copy_ops)

        # Train
        for episode in range(max_episode):
            # Episode Play
            step_count = episode_play(env, episode, mainDQN, replay_buffer)

            if step_count > 10000:      # 이 정도 오래 살렸으면 학습 그만해도 되겠다.
                pass

            # Episode Optimising
            if episode%10 == 1:
                episode_optimize(mainDQN, targetDQN, replay_buffer)
                sess.run(copy_ops)


        # Apply
        state = env.reset()
        reward_sum = 0

        while True:
            env.render()
            action = np.argmax(mainDQN.predict(state))
            next_state, reward, done, info = env.step(action)

            state = next_state
            reward_sum += reward
            if done:
                print('Total score: {}'.format(reward_sum))
                break

main()


'''
Loss: 4.0122
Loss: 10.5154
Loss: 436.877
Loss: 4.29024
Loss: 486.956
Loss: 567.183
Episode 2492 steps 17
Episode 2493 steps 26
Episode 2494 steps 30
Episode 2495 steps 18
Episode 2496 steps 34
Episode 2497 steps 26
Episode 2498 steps 32
Episode 2499 steps 37
Total score: 26.0
학습 잘 안되고 있음
'''
