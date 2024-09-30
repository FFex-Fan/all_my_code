"""
Problem:
    1. if the taxi is near the destination, it will go to the destination directly
    2. the taxi may go in circles (Maybe this phenomenon is decided by reward policy)
"""

import time
from re import findall

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_qtable(n_state, actions):
    table = pd.DataFrame(  # 初始化 Q 表，每个状态-动作对的初始值为0
        np.zeros((n_state, actions)),
        columns=[0, 1, 2, 3, 4, 5]
    )
    return table

def take_action(state, sarsa_table, epsilon, actions):
    # epsilon-greedy 策略选择动作
    state_action = sarsa_table.iloc[state, :]
    if np.random.uniform() < epsilon:
        action = np.random.choice(actions)  # 随机选择动作
    else:
        action = state_action.idxmax()  # 选择当前 Q 值最大的动作
    return action


def get_reward(r, truncated):
    if r:
        return 500
    else:
        if truncated:
            return -100
        else:
            return 1


def train():
    # train
    for epoch in tqdm(range(max_epochs), desc="Training Progress"):
        state, _ = env.reset()  # init state
        done = False
        step_count = 0  # count

        # take action before loop
        action = take_action(state, sarsa_table, epsilon, action_space)

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)

            # reward = get_reward(r, truncated)

            # using next state take action
            next_action = take_action(next_state, sarsa_table, epsilon, action_space)

            q_predict = sarsa_table.iloc[state, action]

            if not (terminated or truncated):
                q_target = reward + gamma * sarsa_table.iloc[next_state, next_action]
            else:
                q_target = reward
                done = True

            TD_error = q_target - q_predict
            # print(TD_error)

            sarsa_table.iloc[state, action] += alpha * TD_error

            state = next_state
            action = next_action

            step_count += 1

        # print(f"Epoch {epoch + 1}: Finished in {step_count} steps.")

    print("Final Sarsa-table:\n", sarsa_table)
    return sarsa_table


def test():
    test_env = gym.make("Taxi-v3", render_mode='human')

    state, _ = test_env.reset()

    action_dim = int(findall(r"\d+\.?\d*", str(test_env.action_space))[0])

    done = False
    step = -1

    while not done:
        step += 1
        action = take_action(state, sarsa_table, epsilon, action_dim)

        ne_state, r, terminated, truncated, info = test_env.step(action)

        done = truncated or terminated
        state = ne_state

        print(f'Test ---- step_num: {step}, action: {action}, reward: {r}, obs: {state}, done: {done}')
        time.sleep(0.1)


if __name__ == '__main__':
    env = gym.make("Taxi-v3")

    state_space = int(findall(r"\d+\.?\d*", str(env.observation_space))[0])
    print("State space size:", state_space)

    action_space = int(findall(r"\d+\.?\d*", str(env.action_space))[0])  # 获取动作维度
    print("Action space size:", action_space)

    max_epochs = 1000  # 训练的 epoch 数量
    alpha = 0.1  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.1  # epsilon greedy 策略参数

    sarsa_table = build_qtable(state_space, action_space)


    train()

    test()


