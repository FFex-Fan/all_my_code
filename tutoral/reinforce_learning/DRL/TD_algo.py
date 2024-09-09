import numpy as np


class TDAgent:
    def __init__(self, state_space_size, alpha=0.1, gamma=0.9):
        """
        初始化TD算法参数和价值函数
        :param state_space_size: 状态空间的大小
        :param alpha: 学习率
        :param gamma: 折扣因子
        """
        self.V = np.zeros(state_space_size)
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state):
        """
        在状态下选择一个动作
        这个例子中，动作是随机选择的
        :param state: 当前状态
        :return: 动作 (a)
        """
        return np.random.choice([0, 1])


    def update(self, state, reward, next_state):
        """
        使用TD(0)更新状态价值
        :param state: 当前状态 (s)
        :param reward: 获得的即时奖励 (r)
        :param next_state: 下一状态 (s')
        """
        td_target = reward + self.gamma * self.V[next_state]
        td_error = td_target - self.V[state]
        self.V[state] += self.alpha * td_error

    def get_value(self, state):
        """
        返回某个状态的价值
        :param state: 状态 (s)
        :return: 状态的价值 V(s)
        """
        return self.V[state]


def get_init():
    """
    模拟一个简单的环境
    这个环境中状态是0, 1, 2, 3, 4，其中状态4是终止状态，获得奖励1，其他状态的奖励为0
    :return: 初始状态
    """
    return np.random.choice([0, 1, 2, 3])


def main():
    state_space_size = 5
    agent = TDAgent(state_space_size)

    num_epochs = 100
    for epoch in range(num_epochs):
        state = get_init()
        while state != 4:
            action = agent.get_action(state)

            next_state = state + 1 if state < 4 else 4
            reward = 1 if next_state == 4 else 0

            agent.update(state, reward, next_state)
            state = next_state

    # 打印学习到的状态价值函数
    print("Learned state values:")
    for state in range(state_space_size):
        print(f"V({state}) = {agent.get_value(state):.4f}")



if __name__ == '__main__':
    main()

