# Objective is to get the cart to the flag.
# for now, let's just move randomly:

import gym
import numpy as np

env = gym.make("MountainCar-v0") # initialize the environment

LEARNING_RATE = 0.1
DISCOUNT = 0.95 # a measure of how important we define the future actions(future rewards) over current actions(current rewards) 
EPISODES = 4000

SHOW_EVERY = 3000


# print(f"env.observation_space.high : {env.observation_space.high}")
# print(f"env.observation_space.low : {env.observation_space.low}")
# print(f"env.action_space.n : {env.action_space.n}")

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # discrete observation size
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# print(f"DISCRETE_OS_SIZE : {DISCRETE_OS_SIZE}")
# print(f"discrete_os_win_size : {discrete_os_win_size}")

# Exploration settings
epsilon = 1 # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#initialize q-table with negative values
# bc the reward is negative until a good event happens
# [20 x 20] (3D)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) 
# print(f"q_table.shape : {q_table.shape}")
# print(q_table) # the values are randomly initialized

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int)) # we use this tuple to look up 3 Q-values for the available actions in the Q-table

for episode in range(EPISODES): 
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset()) # env.reset() gives us the initial state to us
    done = False
    # print(discrete_state)
    # print(f"starting Q-value : {q_table[discrete_state]}")
    # print(f"Maximum Q-value: {np.argmax(q_table[discrete_state])}")

    
    if episode % SHOW_EVERY == 0: 
        render = True
        print(episode)
    else: 
        render = False
        
    while not done:
        
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, _ = env.step(action) # new_state = (position, velocity) of the car
        new_discrete_state = get_discrete_state(new_state)
        
        if episode % SHOW_EVERY == 0:
            env.render()
            
        # If simulation did not end yet after last step - update Q table
        if not done:
            # Maximum possible Q-value in the next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state]) # np.max gives max_values while np.argmax gives the position of max_values 
            # Current Q-value (for current state and performed action)
            current_q = q_table[discrete_state + (action, )] # dicrete_state = (8, 10), action = 0, [discrete_state + (action, )] = (8, 10, 0)
            # And here's our equation for a new Q-value for current state and action
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q # update Q-table with the new Q value
        # Simulation ended (for any reason) - if goal position is achieved - update Q value with reward directly
        elif new_state[0] >= env.goal_position:  # new_state has (position, velocity)
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0 # do nothing : 0 = reward
            
        dicrete_state = new_discrete_state
    
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
        
env.close() 

# every combination of postion of velocity 
# we wanna pick up the maximum Q-values
# initially, the agent explores (does randomly)

"""
| |0|1|---|20|
|--|--|--|--|--|
|C1|0|2| 1|---|1|
"""


"""
Campbell Saint-Vincent
-----------------------
It took me a little bit to understand 
the get_discrete_state() function. 
Essentially what's happening is we have already decided 
to parse the observation values into 20 discrete buckets 
between the highest and lowest possible. 
The discrete_os_window_size is the step between each bucket value. 
By subtracting (state - env.observation_space.low) 
we get some value between the high and low. 
Then, by dividing out discrete_os_window_size , 
we get the i'th bucket the observation falls into. 
By returning .astype(np.int) 
it can be used as an index for the q_table. 
The index is the unique position/velocity bucket.
"""
  