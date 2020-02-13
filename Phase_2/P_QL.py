import numpy as np
import gym
import math

env = gym.make("Pendulum-v0")
observation = env.reset()


def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    
    #determine discretized state size
    Sample_cos_theta = np.around(np.arange(env.observation_space.low[0], env.observation_space.high[0], 0.1), 1)[1:]
    Sample_sin_theta = Sample_cos_theta
    Sample_theta_dot = np.around(np.arange(env.observation_space.low[2], env.observation_space.high[2], 1), 0)[1:]
    Sample_out = np.around(np.arange(-2, 2.2, 0.2), 1)

    bins_angle = len(Sample_cos_theta) + 1
    bins_angle_dot = len(Sample_theta_dot) + 1
    bins_out = len(Sample_out)
    
    #discretize state funciton
    def obs_to_state(observation):
        state_cos_theta = int(np.digitize(observation[0], Sample_cos_theta))
        state_sin_theta = int(np.digitize(observation[1], Sample_sin_theta))
        state_theta_dot = int(np.digitize(observation[2], Sample_theta_dot))
        return (state_cos_theta, state_sin_theta, state_theta_dot)
    
    #initialize q table
    Q = np.zeros(shape=(bins_angle, bins_angle, bins_angle_dot, bins_out), dtype=np.float32)
    
    
    #initialize values to track reward
    reward_list = []
    ave_reward_list = []
    ada_divisor  = 25
    
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
    
    #Run Q-Learning algorithm
    for i in range(episodes):
        done = False
        reward = 0
        tot_reward =0
        observation = env.reset()
        
        #discretize state
        new_state = obs_to_state(observation)

        delta = max(learning, min(1.0, 1.0 - math.log10((i + 1) / ada_divisor)))
        epsilon = max(epsilon, min(1, 1.0 - math.log10((i + 1) / ada_divisor)))
        
        #vvvvvcheck to make sure this is right vvvvv
        for j in range(episodes):
            current_state = new_state
            
            #Determine next action - epsilon greedy strategy
            if np.random.random() < episodes:
                action = np.random.randint(len(Sample_out))
            else:
                action = np.argmax(Q[current_state])
        

            action = Sample_out[action]  # map index to action value

            # get next state and reward
            obs, reward, done, _ = env.step([action])
 
            #discretize new state
            new_state = obs_to_state(observation)
            
            #Adjust Q value for current state
            Q[current_state][int(action)] += delta * (reward + discount * np.max(Q[new_state]) - Q[current_state][int(action)])
            if done:
                break
        
            #update variables
            tot_reward += reward
            current_state = new_state
            
        #Calculate episodic reduction in epsilon
        if epsilon > min_eps:
                epsilon -= reduction
            
        #track rewards
        reward_list.append(tot_reward)
            
            
        if (i+1) % 1000 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
        
            if (i+1) % 5000 == 0:    
                print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
            
    env.close()
        
    return ave_reward_list
        
        
        
        
# Run Q-learning algorithm
X = [0.2, 0.4, 0.6]
Y = [0.2, 0.5, 1.0]
Z = [1.0, 0.8, 0.6]

#a = [1, 2, 3]


for seed in (0, 1, 2):
    observation = env.reset()
    print("This is the seed: ", env.seed(seed))
    print("This is the starting space: ", observation)
    #print("This is the return of the step function: ", env.step(seed))
    print("*******BEGIN GRID SEARCH******************")
    for learning_rate  in X:
        for discount_factor in Y:
            for epsilon in Z:
                rewards = QLearning(env, learning_rate, discount_factor, epsilon, 0, 5000)
                print(learning_rate, discount_factor, epsilon,)
                print("LEARNING RATE | DISCOUNT FACTOR | EPSILON")
