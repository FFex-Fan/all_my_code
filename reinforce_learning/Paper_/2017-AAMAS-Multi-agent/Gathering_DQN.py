import random
import numpy as np
import collections  # Python内置数据结构模块，包含 deque（双端队列）等
import torch.nn.functional as F  # 包含激活函数、损失函数等功能
import matplotlib.pyplot as plt

from Gathering_Params import *


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 创建一个固定大小的队列来存储经验，FIFO 方式

    def add(self, state, action, reward, next_state, done):  # 将一组经验 (state, action, reward, next_state, done) 添加到缓冲区
        self.buffer.append((state, action, reward, next_state, done))  # 将经验元组追加到队列中

    def sample(self, batch_size):  # 随机从缓冲区采样一个批次数据
        transitions = random.sample(self.buffer, batch_size)  # 随机采样 batch_size 个经验
        # 分别解包经验中的状态、动作、奖励、下一个状态和结束标志，并转换为 NumPy 数组
        state, action, reward, next_state, done = zip(*transitions)
        b_s, b_a, b_r, b_ns, b_d = np.array(state), action, reward, np.array(next_state), done
        # 返回一个包含状态、动作、奖励、下一个状态和结束标志的字典
        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }
        return transition_dict  # 返回字典，供训练时使用

    def size(self):  # 返回当前缓冲区中存储的经验数量
        return len(self.buffer)


class Qnet(torch.nn.Module):  # 定义 Q 网络类，继承自 PyTorch 的 Module 基类
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()  # 调用父类的构造函数，初始化网络
        # 定义一个两层的全连接神经网络
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),  # 输入维度为 state_dim，输出维度为 hidden_dim 的全连接层
            torch.nn.ReLU(),  # ReLU 激活函数
            torch.nn.Linear(hidden_dim, action_dim),  # 输入为 hidden_dim，输出为 action_dim 的全连接层
        )

    def forward(self, x):  # 定义前向传播函数
        return self.model(x)  # 将输入 x 通过定义的网络计算出输出


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        # 初始化 Q 网络和目标 Q 网络，分别用于决策和目标计算
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)  # Q 网络，用于决策
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)  # 目标 Q 网络，用于计算目标值
        # Adam 优化器，用于更新 Q 网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的 epsilon 值，用于控制随机探索的概率
        self.target_update = target_update  # 目标网络更新频率，多少次更新后同步目标网络
        self.count = 0  # 用于记录更新次数
        self.device = device  # 计算设备（CPU 或 GPU）

    def take_action(self, state, is_epsilon=True):  # 采用 epsilon-贪婪策略选择动作
        if is_epsilon and np.random.random() < self.epsilon:  # 以 epsilon 概率随机选择动作
            action = np.random.randint(self.action_dim)  # 随机选择动作
        else:
            # 将状态转换为张量并移动到计算设备上
            state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
            # 根据 Q 网络计算出每个动作的 Q 值，选择 Q 值最大的动作
            action = self.q_net(state).argmax().item()
        return action  # 返回选择的动作

    def update(self, transition_dict):  # 更新 Q 网络参数
        # 将经验字典中的状态、动作、奖励、下一个状态和结束标志转换为张量
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 从 Q 网络中提取与执行的动作对应的 Q 值
        q_values = self.q_net(states).gather(1, actions)
        # 从目标 Q 网络中计算下一个状态的最大 Q 值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 计算 TD 目标：奖励 + 折扣因子 * 下一个状态的最大 Q 值 * (1 - done)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 计算均方误差损失函数
        # loss_fn = torch.nn.MSELoss()
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets).to(device))
        self.optimizer.zero_grad()  # 梯度清零
        dqn_loss.backward()  # 反向传播，计算梯度
        self.optimizer.step()  # 更新 Q 网络参数

        # 每隔 target_update 步更新目标 Q 网络的参数
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 将 Q 网络的参数复制到目标 Q 网络
        self.count += 1  # 计数器增加


    def load_model(self, model_path):  # 加载模型
        state_dict = torch.load(model_path)  # 从文件加载 Q 网络的参数
        self.q_net.load_state_dict(state_dict, False)  # 加载参数到 Q 网络中


def cal_reward(_reward, done, step_num, max_step_num):
    reward = _reward  # 初始奖励为环境返回的即时奖励
    if step_num >= max_step_num - 1:  # 如果达到最大步数，则增加额外奖励
        reward += reward * 5  # 增加 5 倍奖励，鼓励完成任务
    elif done:  # 如果提前结束（失败或成功）
        reward = -1  # 设置为负奖励，表示失败
    return reward  # 返回计算后的奖励


def plot_figure(results):
    keys = ['reward', 'success']  # 需要绘制的指标：奖励和成功率
    for k in keys:  # 对每个指标进行绘图
        iteration_list = list(range(len(results['ave_' + k])))  # 获取迭代次数列表
        plt.plot(iteration_list, results['ave_' + k], color='b', label='ave_' + k)  # 平均值曲线
        plt.plot(iteration_list, results['max_' + k], color='r', label='max_' + k)  # 最大值曲线
        plt.plot(iteration_list, results['min_' + k], color='g', label='min_' + k)  # 最小值曲线
        plt.xlabel('Iteration')  # x 轴标签
        plt.ylabel(k)  # y 轴标签
        plt.title('DQN on {}'.format(game_name, k))  # 图标题
        plt.show()  # 显示图像

        figure_path = train_figure_path.replace('.png', '_{}.png'.format(k))  # 图像保存路径
        plt.savefig(figure_path)  # 保存图像到文件


def calc_reward(_reward, done, step_num, max_step_num):
    reward = _reward  # 初始奖励为环境返回的即时奖励
    if step_num >= max_step_num - 1:  # 如果达到最大步数，则增加额外奖励
        reward += reward * 5  # 增加 5 倍奖励，鼓励完成任务
    elif done:  # 如果提前结束（失败或成功）
        reward = -1  # 设置为负奖励，表示失败
    return reward  # 返回计算后的奖励