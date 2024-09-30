import time

from tqdm import tqdm

from Gathering_Params import *
from Gathering_DQN import *
from Gathering_Env import *


def train():
    env = GameEnv()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    replay_buffer1 = ReplayBuffer(buffer_size)
    replay_buffer2 = ReplayBuffer(buffer_size)

    state_dim = 13 * 33 * 3
    action_dim = 8

    agent1 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    agent2 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    results = {}  # 存储训练结果
    for k in ['reward1', 'reward2', 'success']:  # 初始化结果列表
        results['ave_' + k] = []

    for i in range(iteration_num):
        with tqdm(total=episode_num, desc=f'Iteration {i}', position=0) as pbar:
            rewards1, rewards2, successes = [], [], []
            for i_episode in range(episode_num):
                state = env.reset()
                state1 = state
                state2 = state

                episode_reward1 = 0  # 用于累积每个 episode 的奖励
                episode_reward2 = 0

                for step_num in range(max_step_num):
                    action1 = agent1.take_action(state1)
                    action2 = agent2.take_action(state2)

                    # if action1 == 6 or action2 == 6:
                    #     print("train ------ ", action1, action2)

                    next_state, r1, done, info = env.step(action1, action2)
                    r2 = info.get('agent2_reward', 0)

                    reward1 = calc_reward(r1, done, step_num, max_step_num)
                    reward2 = calc_reward(r2, done, step_num, max_step_num)

                    replay_buffer1.add(state1, action1, reward1, next_state, done)
                    replay_buffer2.add(state2, action2, reward2, next_state, done)

                    # 当经验回放缓冲区大小超过最小阈值后，开始训练
                    if replay_buffer1.size() > minimal_size:
                        transition_dict = replay_buffer1.sample(batch_size)  # 从缓冲区采样经验
                        agent1.update(transition_dict)  # 使用采样的经验更新 Q 网络

                    if replay_buffer2.size() > minimal_size:
                        transition_dict = replay_buffer2.sample(batch_size)
                        agent2.update(transition_dict)

                    state1 = next_state
                    state2 = next_state

                    episode_reward1 += reward1  # 累加步骤奖励
                    episode_reward2 += reward2

                    if done: break  # 如果 episode 结束，退出步骤循环

                # 记录每个 episode 的累计奖励和成功情况
                success = 1 if done and step_num < max_step_num - 1 else 0
                successes.append(success)
                rewards1.append(episode_reward1)
                rewards2.append(episode_reward2)

                # 计算平均奖励和成功率
                ave_reward1 = np.mean(rewards1)
                ave_reward2 = np.mean(rewards2)
                ave_success = np.mean(successes)

                # 每 10 个 episode 显示一次进度信息
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (episode_num * i + i_episode + 1),
                        'r1': '%.3f' % ave_reward1,
                        'r2': '%.3f' % ave_reward2,
                        'success': '%.1f' % ave_success,
                    })
                pbar.update(1)  # 在 episode 循环中更新进度条

            # 保存每次迭代的结果
            results['ave_reward1'].append(ave_reward1)
            results['ave_reward2'].append(ave_reward2)
            results['ave_success'].append(ave_success)

        # 保存模型
        torch.save(agent1.q_net.state_dict(), dir_out + '/models/agent1_model.pth')
        torch.save(agent2.q_net.state_dict(), dir_out + '/models/agent2_model.pth')


def test():
    env = GameEnv()

    state_dim = 13 * 33 * 3
    action_dim = 8


    agent1 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    agent2 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    state_dict1 = torch.load(dir_out + '/models/agent1_model.pth')
    agent1.q_net.load_state_dict(state_dict1, strict=False)

    state_dict2 = torch.load(dir_out + '/models/agent2_model.pth')
    agent2.q_net.load_state_dict(state_dict2, strict=False)

    # 重置环境并获取初始状态
    state = env.reset()
    state1 = state
    state2 = state

    step_num = -1
    done = False


    while not done:
        step_num += 1

        # 渲染环境并显示图像
        temp = env.render_env()
        plt.imshow(temp)
        plt.axis('off')  # 隐藏坐标轴
        plt.title(f'Step: {step_num}')
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()

        # 代理选择动作
        action1 = agent1.take_action(state1, is_epsilon=False)
        action2 = agent2.take_action(state2, is_epsilon=False)

        if action1 == 6 or action2 == 6:
            print("test ------ ", action1, action2)

        # 环境执行动作并返回下一个状态和奖励
        next_state, r1, done, info = env.step(action1, action2)
        r2 = info.get('agent2_reward', 0)

        # 更新状态
        state1 = next_state
        state2 = next_state

        # 输出奖励信息
        if r1 or r2:
            print(f"Step: {step_num}\tReward1: {r1}\tReward2: {r2}\tdone: {done}")


        # print(f'Step: {step_num}\tdone: {done}')
        time.sleep(1)

    # 关闭图像窗口
    plt.close()


if __name__ == '__main__':
    train()

    test()