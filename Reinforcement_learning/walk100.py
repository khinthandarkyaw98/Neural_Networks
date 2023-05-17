"""_summary_
Custom env : Agent - Action (Backward, Stay, Forward)
             Goal : 100 steps move
             Observation : location between 0 and 100 steps
             done : the signal to show 100 steps reached by the agent
             reward : (-1, 0, 1)
"""
# Importing requried libraries
import gym
from gym import spaces
import random

# Creating the custom environment
# Custom environment needs to inherit from the abstract class gym.Env
class Walk_Motivation(gym.Env):
  # add the metadata attritbute to your clas
  metadata = {"render_modes": ["human", "rgb_array"],
              "render_fps": 4}

  def __init__(self):
    # define the environmnet's action_space and observation space
    '''
    Box-The argument low specifies the lower bound of each dimension and high specifies the upper bounds
    '''
    # walk from 0 to 100
    self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,)) # 1D array
    
    # action_space are move forward, move backward or stay where you are
    self.action_space = gym.spaces.Discrete(3)
    
    # current state
    self.state = random.randint(0, 20)
    
    # rewards 
    self.reward = 0
    
    
  def get_action_meanings(self, action):
    action_list = {0: "Move Forward",
                   1: "Move backward", 
                   2: "Stay at same postion"}
    return action_list[action]
  
  def step(self, action):
    '''
    defines the logic of your environment when the agent takes an action 
    Accepts the action, computes the state of the environment after applying the action
    '''
    done = False # if terminated 
    info = {}
    
    # setting the state of the environmnet based on agent's action
    # rewarding the agnet for the action
    if action == 0: # Move Forward
      self.state += 10
      self.reward += 1
      
    elif action == 1: # Move backward
      self.state -= 1
      self.reward += -1
      
    elif action == 2: # Stay at the same position
      self.state
      self.reward += 0
      
    # define the completion of the episode
    if self.state >= 101:
      self.reward += 100
      done = True
    self.render(action)
    return self.state, self.reward, done, info

  def render(self, action):
    # Visualize your environment
    print(f"\n Distance Travelled: {self.state}\n Reward Received: {self.reward}")
    print(f"Action taken: {self.get_action_meanings(action)}")  
    print("=" * 20)
    
  def reset(self):
    # reset your environment 
    self.state = random.randint(0, 20)
    self.reward = 0
    return self.state
  
  def close(self):
    # close the environment
    self.state = 0
    self.reward = 0
    
env = Walk_Motivation() # initialize the custom environment class
done = False # set done to False
state = env.reset() # reset the environment
while not done:
  action = env.action_space.sample() # generate random action
  state, reward, done, info = env.step(action) # execute one timestep
env.close()

