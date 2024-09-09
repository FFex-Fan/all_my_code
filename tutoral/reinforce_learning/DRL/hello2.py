import time
import gymnasium as gym
from IPython import display
from matplotlib import pyplot as plt

env = gym.make('MountainCar-v0', render_mode='rgb_array')

obs = env.reset()
num_steps = 1500

for step in range(num_steps):
    action = env.action_space.sample()

    obs, reward, done, truncate, info = env.step(action)

    env.render()

    time.sleep(0.001)

    if done:
        env.reset()


env.close()

print(type(env.observation_space))
print("Upper Bound for Env Observation", env.observation_space.high)
print("Lower Bound for Env Observation", env.observation_space.low)

env.step(2)
print("It works")

# env.step(4)
# print("It works")
