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

# env_name = 'Breakout-v0'
# env_name = 'SpaceInvaders-v0'


def run_training_and_testing_experiment(experiment_configurations: []):
    for experiment_configuration in experiment_configurations:
        env_name = experiment_configuration['env_name']
        training_episodes = experiment_configuration['training_episodes']
        testing_episodes = experiment_configuration['testing_episodes']
        learning_rate = experiment_configuration['learning_rate']
        discount_factor = experiment_configuration['discount_factor']
        epsilon = experiment_configuration['epsilon']
        experiment_name = experiment_configuration['experiment_name']

        config_identifier = experiment_name + '_' + env_name + '_' + str(training_episodes) + 'training_episodes_' \
                            + str(testing_episodes) + 'testing_episodes_' + str(learning_rate) + 'lr_' \
                            + str(discount_factor) + 'discount'

        rl.checkpoint_base_dir = 'results/' + now + '_checkpoints_tutorial16/'

        rl.update_paths(env_name=env_name)

        agent = rl.Agent(env_name=env_name,
                         training=True,
                         render=True,
                         use_logging=True,
                         config_identifier=config_identifier,
                         epsilon=epsilon,
                         starting_learning_rate=learning_rate,
                         discount_factor=discount_factor)

        model = agent.model

        # replay_memory = agent.replay_memory

        # we are training here:
        agent.run(num_episodes=training_episodes)

        # log_q_values = rl.LogQValues()
        # log_reward = rl.LogReward()
        log_q_values = agent.log_q_values
        log_reward = agent.log_reward

        log_q_values.read()
        log_reward.read()

        figure_file_name_end = '_' + config_identifier + '.png'

        plt.plot(log_reward.count_states, log_reward.episode, label='Episode Reward')
        plt.plot(log_reward.count_states, log_reward.mean, label='Mean of 30 episodes')
        plt.xlabel('State-Count for Game Environment')
        plt.legend()
        # plt.show()
        plt.savefig(rl.checkpoint_base_dir + 'training-progress-reward' + figure_file_name_end)
        plt.close()

        plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
        plt.xlabel('State-Count for Game Environment')
        plt.legend()
        # plt.show()
        plt.savefig(rl.checkpoint_base_dir + 'training-progress-Q-values' + figure_file_name_end)
        plt.close()

        agent.epsilon_greedy.epsilon_testing

        agent.training = False

        agent.reset_episode_rewards()

        # faster to not render the screen, toggle these commented lines to see teh video output
        # agent.render = True
        agent.render = False

        # testing the agent occurs here
        agent.run(num_episodes=testing_episodes)

        rewards = agent.episode_rewards
        reward_string = "Rewards for {0} episodes:".format(len(rewards)) + "\n"
        reward_string += "- Min:   " + str(np.min(rewards)) + "\n"
        reward_string += "- Mean:  " + str(np.mean(rewards)) + "\n"
        reward_string += "- Max:   " + str(np.max(rewards)) + "\n"
        reward_string += "- Stdev: " + str(np.std(rewards)) + "\n"
        print(reward_string)
        with open(rl.checkpoint_base_dir + "Rewards_" + config_identifier + ".txt", "w") as text_file:
            text_file.write(reward_string)

        plt.hist(rewards, bins=testing_episodes)
        plt.savefig(rl.checkpoint_base_dir + 'testing_rewards' + figure_file_name_end)
        plt.close()

        max_reward_state = np.argmax(agent.replay_memory.rewards)

        print_q_values(agent, max_reward_state)


def print_q_values(agent, idx):
    """Print Q-values and actions from the replay-memory at the given index."""

    # Get the Q-values and action from the replay-memory.
    q_values = agent.replay_memory.q_values[idx]
    action = agent.replay_memory.actions[idx]

    print("Action:     Q-Value:")
    print("====================")

    # Print all the actions and their Q-values.
    for i, q_value in enumerate(q_values):
        # Used to display which action was taken.
        if i == action:
            action_taken = "(Action Taken)"
        else:
            action_taken = ""

        # Text-name of the action.
        action_name = agent.get_action_name(i)

        print("{0:12}{1:.3f} {2}".format(action_name, q_value,
                                         action_taken))

    # Newline.
    print()


if __name__ == '__main__':
    print('starting')
    num_training_episodes = 500
    num_testing_episodes = 50
    default_learning_rate = 0.001
    default_discount_factor = 0.97
    default_epsilon = 0.1

    # experiments
    # original_tutorial_experiment = {
    #     'env_name': 'Breakout-v0',
    #     'training_episodes': num_training_episodes,
    #     'testing_episodes': num_testing_episodes,
    #     'learning_rate': default_learning_rate,
    #     'discount_factor': default_discount_factor,
    #     'epsilon': default_epsilon,
    #     'experiment_name': 'original_tutorial_experiment'
    # }

    pong_default = {  # pong_small_learning_rate_large_discount
        'env_name': 'Pong-v0',
        'training_episodes': num_training_episodes,
        'testing_episodes': num_testing_episodes,
        'learning_rate': default_learning_rate,
        'discount_factor': default_discount_factor,
        'epsilon': default_epsilon,
        'experiment_name': 'pong_default'
    }

    pong_large_learning_rate = {
        'env_name': 'Pong-v0',
        'training_episodes': num_training_episodes,
        'testing_episodes': num_testing_episodes,
        'learning_rate': .5,
        'discount_factor': default_discount_factor,
        'epsilon': default_epsilon,
        'experiment_name': 'pong_large_learning_rate'
    }

    # pong_large_learning_rate_small_discount = {
    #     'env_name': 'Pong-v0',
    #     'training_episodes': num_training_episodes,
    #     'testing_episodes': num_testing_episodes,
    #     'learning_rate': .5,
    #     'discount_factor': .1,
    #     'epsilon': default_epsilon,
    #     'experiment_name': 'pong_large_learning_rate_small_discount'
    # }

    pong_small_small_discount = {
        'env_name': 'Pong-v0',
        'training_episodes': num_training_episodes,
        'testing_episodes': num_testing_episodes,
        'learning_rate': default_learning_rate,
        'discount_factor': .1,
        'epsilon': default_epsilon,
        'experiment_name': 'pong_small_small_discount'
    }

    pong_large_epsilon = {
        'env_name': 'Pong-v0',
        'training_episodes': num_training_episodes,
        'testing_episodes': num_testing_episodes,
        'learning_rate': default_learning_rate,
        'discount_factor': default_discount_factor,
        'epsilon': 0.9,
        'experiment_name': 'pong_large_epsilon'
    }

    training_and_testing_experiment_configs = [
        pong_default,
        pong_large_learning_rate,
        pong_small_small_discount,
        pong_large_epsilon
    ]
    run_training_and_testing_experiment(training_and_testing_experiment_configs)
