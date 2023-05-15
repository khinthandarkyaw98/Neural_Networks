import gym

env = gym.make("MountainCar-v0")
env.reset()

done = False

while not done:
    action = 2 # 3 actions : 0 push left, 1 do nth, 2 push right
    env.step(action) # new_state = (position, velocity) of the car
    env.render()
    
env.close()
  