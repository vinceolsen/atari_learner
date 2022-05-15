import matplotlib.pyplot as plt
import gym
import numpy as np
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import reinforcement_learning as rl

import ale_py
import time


now = str(int(time.time()))

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

# TensorFlow
tf.__version__

# OpenAI Gym
gym.__version__

# env = gym.make("Breakout-v0")

env_name = 'Breakout-v0'
# env_name = 'SpaceInvaders-v0'

rl.checkpoint_base_dir = 'results/' + now + '_checkpoints_tutorial16/'

rl.update_paths(env_name=env_name)

agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=True,
                 use_logging=True)

model = agent.model

# replay_memory = agent.replay_memory

agent.run(num_episodes=3)

# log_q_values = rl.LogQValues()
# log_reward = rl.LogReward()
log_q_values = agent.log_q_values
log_reward = agent.log_reward

log_q_values.read()
log_reward.read()

plt.plot(log_reward.count_states, log_reward.episode, label='Episode Reward')
plt.plot(log_reward.count_states, log_reward.mean, label='Mean of 30 episodes')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()

plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()

agent.epsilon_greedy.epsilon_testing


agent.training = False

agent.reset_episode_rewards()

agent.render = True

agent.run(num_episodes=1)

agent.reset_episode_rewards()

agent.render = False

agent.run(num_episodes=30)

if __name__ == '__main__':
    print('starting')
