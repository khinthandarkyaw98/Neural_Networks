import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

print(f"env.observation_space.high : {env.observation_space.high}")
print(f"env.observation_space.low : {env.observation_space.low}")
print(f"env.action_space.n : {env.action_space.n}")

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # discrete observation size
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(f"DISCRETE_OS_SIZE : {DISCRETE_OS_SIZE}")
print(f"discrete_os_win_size : {discrete_os_win_size}")

#initialize q-table with negative values
# bc the reward is negative until a good event happens
# [20 x 20] (3D)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) 
print(f"q_table.shape : {q_table.shape}")
print(q_table) # the values are randomly initialized

""" done = False

while not done:
    action = 2 # 3 actions : 0 push left, 1 do nth, 2 push right
    new_state, reward, done, _ = env.step(action) # new_state = (position, velocity) of the car
    print(reward, new_state) 
    env.render()
    
env.close() """

# every combination of postion of velocity 
# we wanna pick up the maximum Q-values
# initially, the agent explores (does randomly)

"""
| |0|1|---|20|
|--|--|--|--|--|
|C1|0|2| 1|---|1|
"""



  