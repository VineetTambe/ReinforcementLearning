
import pickle
import gymnasium as gym
import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
import time

file = open("q_table25000.obj",'rb')
q_table = pickle.load(file)
file.close()

# Show results of training

new_env = gym.make("LunarLander-v2", render_mode="human")
observation, info = new_env.reset(seed = 50)


DISCRETIZATION = 12

DISCRETE_OBSERVATION_SPACE_SIZE = [DISCRETIZATION] * len(observation[:-2])
discrete_os_win_size = (new_env.observation_space.high[:-2] - new_env.observation_space.low[:-2]) / DISCRETE_OBSERVATION_SPACE_SIZE

def get_discrete_state(state):
    descrete_state = ((state[:-2] - new_env.observation_space.low[:-2]) / discrete_os_win_size)
    descrete_state =  np.clip(descrete_state, 0 , DISCRETIZATION-1)
    # print(type(descrete_state))
    # print(tuple(descrete_state.astype(int).clip(0, DISCRETE_OBSERVATION_SPACE_SIZE-1)))
    return tuple(descrete_state.astype(int))

terminated = False
truncated = False
discrete_state = get_discrete_state(observation)
while not (terminated or truncated):
    # print(q_table[discrete_state])
    action = np.argmax(q_table[discrete_state])
    # print(action)
    observation, reward, terminated, truncated, _ = new_env.step(action)    
    new_discrete_state = get_discrete_state(observation)        
    discrete_state = new_discrete_state
    time.sleep(0.1)
    print(reward)
new_env.close()