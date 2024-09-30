import torch

game_name = 'Gathering'
method_name = 'DQN'

# buffer
buffer_size = 10000
minimal_size = 500
batch_size = 64

# model
hidden_dim = 32
device = torch.device("cpu")

target_update = 10
epsilon = 0.01
gamma = 0.98
lr = 2e-3

# == iteration
iteration_num = 3  # 组数
episode_num = 100  # 每组的步骤数

max_step_num = 200  # 单局的最大步长

# == path
dir_data = './data/' + method_name
dir_out = './output/' + method_name + '_' + game_name
dir_models = dir_out + '/models'
dir_figures = dir_out + '/figures'
model_path = dir_out + '/models/lastest.pt'
train_result_path = dir_out + '/train_result.csv'
train_figure_path = dir_out + '/train_figure.png'
test_result_path = dir_out + '/test_result.csv'
