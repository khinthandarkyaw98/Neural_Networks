import gym 

episodes = 1000
env = gym.make('MountainCar-v0') 
observation, info = env.reset()
for i in range(episodes):
  observation, reward, terminated, _ = env.step(env.action_space.sample()) # step: gives state transition
  env.render()
  
if terminated: # the end of the episodes
  print(f"Objective achieved at episode {i}")
  env.reset()
  
env.close()