import numpy as np
import gym
import matplotlib.pyplot as plt
env_v = gym.make("MountainCar-v0", render_mode="human")
env_v.reset()
print("The State Space: ", env_v.observation_space)
print("The Action Space: ", env_v.action_space)
l_value = env_v.observation_space.low
h_value = env_v.observation_space.high
print("The Bound Value of  X Axis: ", l_value[0], h_value[0])
print("The Bound Value of  Y Axis: ", l_value[1], h_value[1])
# Define Query-learning function
def Query_Learning(env_v, learn_value, val_of_disc, eps_val, min_no_of_eps, no_of_episodes):
    # Determine size of discretized state space
    no_of_state = (env_v.observation_space.high - env_v.observation_space.low) * \
                 np.array([10, 100])
    no_of_state = np.round(no_of_state, 0).astype(int) + 1
    # Initialize Query table
    Query_Table = np.random.uniform(low=-1, high=1,
                          size=(no_of_state[0], no_of_state[1],
                                env_v.action_space.n))
    # Initializing the variables to track rewards
    reward_list = []
    ave_reward_list = []
    # Calculating episodic reduction in eps_val
    reduction = (eps_val - min_no_of_eps) / no_of_episodes
    # Run Query learn_value algorithm
    for i in range(no_of_episodes):
        # Initialize parameters
        done = False
        total_no_of_rewards, reward = 0, 0
        state = env_v.reset()
        adj_state_val = (state[0]- env_v.observation_space.low) * np.array([10, 100])
        adj_state_val = np.round(adj_state_val, 0).astype(int)
        while done != True:
            # Rendering environment ( for last 5 no_of_episodes )
            if i >= (no_of_episodes - 10):
                env_v.render()
            # Determining the next act_val
            if np.random.random() < 1 - eps_val:
                act_val = np.argmax(Query_Table[adj_state_val[0], adj_state_val[1]])
            else:
                act_val = np.random.randint(0, env_v.action_space.n)
            # Getting to the next state and reward
            next_state, reward, done, info,v = env_v.step(act_val)
            adj_next_state = (next_state - env_v.observation_space.low) * np.array([10, 100])
            adj_next_state = np.round(adj_next_state, 0).astype(int)
            # Allowing for terminal states
            if done and next_state[0] >= 0.5:
                Query_Table[adj_state_val[0], adj_state_val[1], act_val] = reward
            # Adjust Query value for current state
            else:
                delta = learn_value * (reward +
                                    val_of_disc * np.max(Query_Table[adj_next_state[0],
                                                        adj_next_state[1]]) -
                                    Query_Table[adj_state_val[0], adj_state_val[1], act_val])
                Query_Table[adj_state_val[0], adj_state_val[1], act_val] += delta
            total_no_of_rewards += reward
            adj_state_val = adj_next_state
        if eps_val > min_no_of_eps:
            eps_val -= reduction
        # Tracking of the  rewards
        reward_list.append(total_no_of_rewards)

        if (i + 1) % 100 == 0:
            reward_val_av = np.mean(reward_list)
            ave_reward_list.append(reward_val_av)
            reward_list = []

        if (i + 1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(i + 1, reward_val_av))
    env_v.close()
    return ave_reward_list
# Executing Query-learn_value algorithm
rewards = Query_Learning(env_v, 0.2, 0.9, 0.8, 0, 5000)
# Plotting the  Rewards
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('The Episodes')
plt.ylabel('The Average Reward Value')
plt.title('The Average Reward vs The Episodes')
plt.savefig('The rewards image.jpg')
plt.close()