
from os import stat
import numpy as np
import torch
import torch.nn as nn

import gym
import gym_2048

from experience_replay import ExperienceReply
from agent import DQNAgent
from dueling_dqn import DuelingDQN

from torch.utils.tensorboard import SummaryWriter

def adjust_reward(state, reward):
    
    max_value_position = np.argmax(state)
    
    if max_value_position in [0, 3, 12, 15] and reward > 0: 
        reward = reward * 2
    
    elif max_value_position in [1, 2, 4, 7, 8, 11] and reward > 0:
        reward = reward * 1.5
    
    elif max_value_position in [5, 6, 9, 10] and reward == 0:
        reward -= 10
    
    return reward


def state_processing(state):
    # reshape state from 4x4 to 1x16
    state = state.copy().reshape(1, 16) + np.random.rand(1, 16)/10.0
    # convert from numpy to torch
    state = torch.from_numpy(state).float()
    # log_2
    state = torch.log2(state+1)
    return state


def test_agent(agent, episode_num=100):

    test_env = gym.make("2048-v0")
    episode_rewards = []

    for episode in range(episode_num):
        episode_reward = 0
        
        done = False
        move_steps = 0
        
        # initial state
        _state = test_env.reset()    
        state = state_processing(_state)

        while (not done):
            action = agent.get_action(state)
            _next_state, reward, done, _ = test_env.step(action)
            move_steps += 1
            episode_reward += reward

            _next_state = state_processing(_next_state)
            
            if done:
                episode_rewards.append(episode_reward)
                print('**'*25)
                print(f'Validation Mode @Episode. {episode}')
                print(f'Rewards: {episode_reward},  Total moves: {move_steps}')
                print('--'*25)
                test_env.render()
                print('--'*25)
                break

    # reutnr average reward.
    return sum(episode_rewards)/len(episode_rewards)
        

def minibatch_train(env, agent, max_episodes, batch_size, writer=None):
    episode_rewards = []

    for train_ep in range(max_episodes):

        episode_reward = 0
        move_steps = 0
        done = False

        _state = env.reset()
        state = state_processing(_state)

        while (not done):
            action = agent.get_action(state)
            _next_state, reward, done, _ = env.step(action)
            move_steps += 1
            episode_reward += reward
            
            # adjusting reward, but doesn't change the episode_reward.
            reward = adjust_reward(_next_state, reward)

            next_state = state_processing(_next_state)
            agent.experience_replay.push(state, action, reward, next_state, done)   

            if len(agent.experience_replay) > batch_size:
                agent.update(batch_size)

            if done:
                episode_rewards.append(episode_reward)
                print('**'*25)
                print(f'Training Mode @Episode. {train_ep}')
                print(f'Rewards: {episode_reward},  Total moves: {move_steps}')
                print('--'*25)
                env.render()
                print('--'*25)
                break
            
            state = next_state
        
            if agent.epsilon > 0.1:  
               agent.epsilon -= (1/max_episodes)
        
        
        # Every 200 episode, copy the current main model parameters of the Q network to the target model
        if train_ep % 200 == 0:

            # every 200 ep, testing the model 
            val_avg_reward = test_agent(agent, 20)
            
            agent.target_model.load_state_dict(agent.model.state_dict())

            if writer:
                writer.add_scalar('Valid Avg Reward', val_avg_reward, train_ep)

        if train_ep % 1000 == 0:
            agent.save_checkpoint(train_ep)


    return episode_reward

if __name__ == '__main__':
    
    MAX_EPISODES = 100000
    BATCH_SIZE = 200
    MAMORY_SIZE = 5000

    # creating 2048 env
    env = gym.make("2048-v0")

    # main model
    main_model = DuelingDQN(16, 4)
    
    # target model
    target_model = DuelingDQN(16, 4)
    target_model.load_state_dict(main_model.state_dict())

    # optimizer for main model
    optimizer = torch.optim.Adam(main_model.parameters())

    # experience replay
    experience_replay = ExperienceReply(max_size=MAMORY_SIZE)

    # MSE loss function 
    loss_function = nn.MSELoss()

    # agent
    agent = DQNAgent(env, main_model, target_model, optimizer, loss_function, experience_replay)

    # tf board
    tbWriter = SummaryWriter(f'runs/2048game/')

    # training with mini-batch
    episode_rewards = minibatch_train(env, agent, MAX_EPISODES, BATCH_SIZE, writer=tbWriter)