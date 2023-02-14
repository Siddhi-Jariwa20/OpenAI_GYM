import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0", render_mode="human")

#printing the value of observation and action space
print("The State Space: ", env.observation_space)
print("The Action Space: ", env.action_space)

#getting the high and low value
l_value = env.observation_space.low
h_value = env.observation_space.high

#printing the Bounds for X and Y axis respectively
print("The Bound Value for  X Axis: ", l_value[0], h_value[0])
print("The Bound Value for Y Axis: ", l_value[1], h_value[1])
#resetting the environment
env.reset()
# A loop for taking a random action for 60000 times
for _ in range(6000):
    env.render()
    env.step(env.action_space.sample())
env.close()
# Executing Query-learning algorithm
tot_rewards = Query_Learning(env, 0.2, 0.9, 0.8, 0, 6000)

# Plotting the rewards
plt.plot(100 * (np.arange(len(tot_rewards)) + 1), tot_rewards)
plt.xlabel('Events')
plt.ylabel('The Value of Average Reward')
plt.title('The Average Reward vs Events')
plt.savefig('rewards_image.jpg')
plt.close()