import gym
import math
import numpy as np

x_range = [-1.2, 0.6]
v_range = [-0.07, 0.07]

def choose_action(state, q_table, action_space, epsilon):
    if np.random.random_sample() < epsilon:  # random select an action with probability epsilon
        return action_space.sample()  # random select an action
    else:  # Choose an action according to the Q table
        return np.argmax(q_table[state])


def get_state(observation, n_buckets):
    state = [0] * len(observation) # 2
    """
    TO-DO: convert continuous feature values to discrete buckets
    """
    x, v = observation
    x_b, v_b = n_buckets
    state[0] = int(((x - x_range[0]) / (x_range[1] - x_range[0])) * x_b)
    state[1] = int(((v - v_range[0]) / (v_range[1] - v_range[0])) * v_b)
    return tuple(state)


def init_qtable():
    """ TO-DO Decide number of buckets for 4 features """
    n_buckets = (16, 16)
    n_actions = env.action_space.n  # Number of actions
    q_table = np.zeros(n_buckets + (n_actions,))

    return n_buckets, n_actions, q_table


def get_epsilon(i):
    """ TO-DO: Please implement your method to choose epsilon """
    return np.exp(-i)


def get_lr(i):
    """ TO-DO: Please implement your method to choose learning rate """
    return np.exp(-i/5)


def get_gamma(i):
    """ TO-DO: Please implement your method to choose reward discount factor """
    return 0.7


# Main
env = gym.make("MountainCar-v0")  # Select an environment


sum_rewards = 0
n_success = 0
f = open('mountaincar_qtable_log.txt', 'w')

n_buckets, n_actions, q_table = init_qtable()

# Q-learning
for i_episode in range(1000):
    total_reward = 0
    epsilon = get_epsilon(i_episode)
    lr = get_lr(i_episode)
    observation = env.reset()  # reset the environment and return the default observation
    state = get_state(observation, n_buckets)

    for t in range(1000):
        if (i_episode == 999):
            env.render()  # Show the environment in GUI

        action = choose_action(state, q_table, env.action_space, epsilon)
        observation, reward, done, info = env.step(action)  # Take an action on current environment
        total_reward += reward
        sum_rewards += reward

        next_state = get_state(observation, n_buckets)

        q_next_max = np.amax(q_table[next_state])  # The expected max reward for next state
        q_table[state + (action,)] += lr * (reward + get_gamma(i_episode) *
                                            q_next_max - q_table[state + (action,)])  # Update Q table
        state = next_state
        if done:  # Episode terminate
            break

    if total_reward > -200:
        n_success += 1
    if (i_episode + 1) % 100 == 0:
        avg_rewards = sum_rewards / (i_episode + 1)
        log = '{:<2d} successes in {:4d} episodes, avg = {}'.format(n_success, i_episode + 1, avg_rewards)
        print(log)
        f.write(log + '\n')

    print('Episode {:4d}, total rewards {}'.format(i_episode + 1, total_reward))

f.close()
env.close()
