import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time 

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=0)

# print (len(observation))

LEARNING_RATE = 0.1 
DISCOUNT = 0.9
EPISODES = 25000
SHOW_EVERY = 2000

DISCRETIZATION = 12

DISCRETE_OBSERVATION_SPACE_SIZE = [DISCRETIZATION] * len(observation[:-2])
discrete_os_win_size = (env.observation_space.high[:-2] - env.observation_space.low[:-2]) / DISCRETE_OBSERVATION_SPACE_SIZE   

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# print(DISCRETE_OBSERVATION_SPACE_SIZE)
# print(discrete_os_win_size)
# print(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n])
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n]))

print("q_table.shape", q_table.shape)

ep_rewqards = []
aggre_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state):
    descrete_state = ((state[:-2] - env.observation_space.low[:-2]) / discrete_os_win_size)
    descrete_state =  np.clip(descrete_state, 0 , DISCRETIZATION-1)
    return tuple(descrete_state.astype(int))

for episode in tqdm(range(EPISODES)):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        # print(episode)
        render = True
    else:
        render = False


    obs, info = env.reset()

    discrete_state = get_discrete_state(obs)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # print("here")
        action = 2

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        observation, reward, terminated, truncated, _ = env.step(action)
        
        episode_reward += reward
        

        # print(observation)

        new_discrete_state = get_discrete_state(observation)

        if not (terminated or truncated):

            # print(new_discrete_state)

            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            
            # exponential decay
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action, )] = new_q

        elif (observation[0],observation[2]) == (0,0):
            print(f"Landed successfully on episode {episode}!")
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
        # print(reward)
        # env.render()

    ep_rewqards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewqards[-SHOW_EVERY:]) / len(ep_rewqards[-SHOW_EVERY:])
        aggre_ep_rewards['ep'].append(episode)
        aggre_ep_rewards['avg'].append(average_reward)
        aggre_ep_rewards['min'].append(min(ep_rewqards[-SHOW_EVERY:]))
        aggre_ep_rewards['max'].append(max(ep_rewqards[-SHOW_EVERY:]))
        print(f"Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}")
    
env.close()
# plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['avg'], label="avg")
# plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['min'], label="min")
# plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['max'], label="max")
# plt.legend(loc=4)
# plt.show()

import pickle

filename = "q_table" + str(EPISODES) + ".obj"
filehandler = open(filename,"wb")
pickle.dump(q_table,filehandler)
filehandler.close()