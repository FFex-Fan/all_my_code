import numpy as np

from Gathering_Params import *
from Gathering_DQN import *
from Gathering_Env import *

def get_distance(x1, y1, x2, y2):
    return np.abs(x1 - x2) + np.abs(y1 - y2)

# agent2 take greedy policy
def greedy_polity(env, agent): # 先不考虑 Agent 朝向问题
    all_apple_pos = env.get_all_foods()
    my_pos = (agent.x, agent.y)

    closest_dis = 1e9
    closest_apple = None
    for apple in all_apple_pos:
        cur_dis = get_distance(my_pos[0], my_pos[1], apple[0], apple[1])
        if cur_dis < closest_dis:
            closest_apple = apple
            closest_dis = cur_dis

    direction = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    for i in direction:
        dx = my_pos[0] + i[0]
        dy = my_pos[1] + i[1]

        new_dis = get_distance(dx, dy, closest_apple[0], closest_apple[1])

        if new_dis < closest_dis: # 每移动一步，距离只有可能增加或者是减少
            return i

    return 0, 0





def train():
    env = GameEnv()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = 13 * 33 * 3
    action_dim = 8

    replay_buffer = ReplayBuffer(buffer_size)




if __name__ == '__main__':
    train()