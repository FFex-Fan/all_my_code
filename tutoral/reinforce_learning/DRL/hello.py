import gymnasium as gym
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0", render_mode="rgb_array")

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space

print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))


# 重置环境，并查看结果（注意：虽然是重置，但也是初始化，即：env 开始使用的时候需要执行）
obs = env.reset()
print("The initial observation is {}".format(obs))


# 整个动作空间中采用随机的动作
random_action = env.action_space.sample()

# 采取行动，获取新的观察空间
new_obs, reward, done, truncated, info = env.step(random_action)
print("The new observation is {}".format(new_obs))


env_screen = env.render()
plt.imshow(env_screen)


plt.show()


