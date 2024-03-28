import numpy as np
import gym
import gym.wrappers
import time

def get_action(env, qtable, state, epsilon):
    policy = np.random.uniform(0,1)
    if policy > epsilon:
        return np.argmax(qtable[state,:])   
    else:
         return env.action_space.sample()

def Q_learning():
    # Source: https://www.gymlibrary.dev/environments/toy_text/taxi/
    env = gym.make('Taxi-v3')
    qtable = np.zeros((500, 6))

    episodes = 1000
    max_timesteps = 50
    epsilon = 1
    epsilon_decay_rate= 0.005
    learning_rate = 0.9
    discount_factor = 0.8
    

    for each in range(episodes):
        state = env.reset()
        done = False
        state = state[0]
        step = 0
        while step < max_timesteps:
            action = get_action(env, qtable, state, epsilon)

            state2, reward, terminated, done, info = env.step(action)

            qtable[state,action] = qtable[state,action] + learning_rate * \
            (reward + discount_factor * np.max(qtable[state2,:])-qtable[state,action])

            state = state2
            step += 1

            if done == True or terminated == True:
                break
        epsilon = np.exp(-epsilon_decay_rate*each)

    state = env.reset()
    done = False
    rewards = 0
    state = state[0]
    a = env.unwrapped
    a.render_mode = "human"
    step = 0
    while step < max_timesteps:
        action = np.argmax(qtable[state,:])
        state2, reward, done, terminated, info = env.step(action)
        rewards += reward
        state = state2
        time.sleep(2)
        if done == True or terminated == True:
            break

    env.close()

def SARSA():
    # Source: https://www.gymlibrary.dev/environments/toy_text/taxi/
    env = gym.make('Taxi-v3')
    qtable = np.zeros((500, 6))

    episodes = 5000
    max_timesteps = 50
    epsilon = .4
    epsilon_decay_rate= 0.005
    learning_rate = 0.1
    discount_factor = 0.95
    

    for each in range(episodes):
        state = env.reset()
        done = False
        state = state[0]
        step = 0
        action = get_action(env, qtable, state, epsilon)
        while step < max_timesteps:
            

            state2, reward, terminated, done, info = env.step(action)
            action2 = get_action(env, qtable, state2, epsilon)
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_factor * qtable[state2,action2]-qtable[state,action])

            state = state2
            action = action2
            step += 1

            if done == True or terminated == True:
                break
        epsilon = np.exp(-epsilon_decay_rate*each)

    state = env.reset()
    done = False
    rewards = 0
    state = state[0]
    a = env.unwrapped
    a.render_mode = "human"
    step = 0
    while step < max_timesteps:
        action = np.argmax(qtable[state,:])
        state2, reward, done, terminated, info = env.step(action)
        rewards += reward
        state = state2
        time.sleep(2)
        if done == True or terminated == True:
            break

    env.close()

def Double_Q_get_action(env, qtableA, qtableB, state, epsilon):
    policy = np.random.uniform(0,1)
    if policy > epsilon:
        tableSums = []
        x=0
        for x in range(6):
            tableSums.append(qtableA[state,x]+qtableB[state,x])
        
        maxVal = max(tableSums)
        maxList = []
        for x in range(6):
            if tableSums[x] == maxVal:
                maxList.append(x)
        action = np.random.choice(maxList)
        return action
    else:
         return env.action_space.sample()
        
def get_table(qtable, state):
        return np.argmax(qtable[state])

def Double_Q_learning():
    # Source: https://www.gymlibrary.dev/environments/toy_text/taxi/
    env = gym.make('Taxi-v3')
    qtableA = np.zeros((500, 6))
    qtableB = np.zeros((500, 6))
    episodes = 10000
    max_timesteps = 50
    epsilon = 1
    epsilon_decay_rate= 0.005
    learning_rate = 0.1
    discount_factor = 1.0
    reward_data = []
    reward = 0
    
    for each in range(episodes):
        state = env.reset()
        state = state[0]
        done = False
        step = 0
        reward_data.append(reward)
        while step < max_timesteps:
            action = Double_Q_get_action(env, qtableA, qtableB, state, epsilon)
            state2, reward, done, terminated, info = env.step(action)
            randomUpdate = np.random.uniform(0,1)
            if randomUpdate > .5:
                action2 = get_table(qtableA, state2)
                qtableA[state,action] = qtableA[state,action] + learning_rate * (reward + discount_factor * qtableB[state2,action2]-qtableA[state,action])
            else:
                action2 = get_table(qtableB, state2)
                qtableB[state,action] = qtableB[state,action] + learning_rate * (reward + discount_factor * qtableA[state2,action2]-qtableB[state,action])

            state = state2
            step += 1

            if done == True:
                break
        epsilon = np.exp(-epsilon_decay_rate*each)
    
    state = env.reset()
    done = False
    rewards = 0
    state = state[0]
    a = env.unwrapped
    a.render_mode = "human"
    step = 0
    while step < max_timesteps:
        action = np.argmax(qtableA[state,:]+qtableB[state,:])
        state2, reward, done, terminated, info = env.step(action)
        rewards += reward
        state = state2
        time.sleep(2)
        if done == True or terminated == True:
            break

    env.close()
if __name__ == "__main__":
    print("gym version(0.26.1) or above required to render simulation")
    print("Creating simulation please wait...")
    print("Q-Learning Simulation")
    Q_learning()
    print("SARSA Simulation")
    SARSA()
    print("DOUBLE-Q-Learning Simulation")
    Double_Q_learning()