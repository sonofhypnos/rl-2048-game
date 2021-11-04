
import torch
import numpy as np
import os

class DQNAgent:
    
    def __init__(self, env, model, target_model, optimizer, loss_function, experience_replay, learning_rate=3e-4, gamma=0.99):
        
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 0.999
        self.experience_replay = experience_replay
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.optimizer = optimizer
        self.loss_function = loss_function

    @staticmethod
    def scaling_rewards(rewards, interval_max=1, interval_min=-1):
        '''
            Scaling reward's range to (interval_min, interval_max)
        '''
        m = (interval_max - interval_min) / (rewards.max() - rewards.min())
        b = interval_min - m * rewards.min()
        return m * rewards + b
        
    @staticmethod
    def epsilon_greedy_policy(q_value, epsilon): 
        '''
            when random number greater than epsilon, choose action from q value else explosion.
        '''
        if(np.random.randn() > epsilon):
            _action = torch.argmax(q_value)
        else:
            _action = np.random.randint(0, 4)
        return _action

    def get_action(self, state):

        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model(state)
        
        # choose action from policy
        action = DQNAgent.epsilon_greedy_policy(qvals, self.epsilon)

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        rewards = DQNAgent.scaling_rewards(rewards)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        # get current q-value from main model.
        curr_Q = self.model(states)
        predict = curr_Q.gather(dim=1, index = actions.long().unsqueeze(dim=1)).squeeze() 
        
        with torch.no_grad(): 
            # get next q-value from target model.
            next_Q = self.target_model(next_states)
        
        max_next_Q = torch.max(next_Q, dim=1)[0]
        expected_Q = rewards + self.gamma * max_next_Q
        loss = self.loss_function(predict, expected_Q.detach())
        
        return loss

    def save_checkpoint(self, epoch, PATH='./checkpoint'):
        save_path = PATH + os.sep + f'ep-{epoch}-checkpoint.pk'
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        torch.save(checkpoint_dict, save_path)

    def load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def update(self, batch_size):

        batch = self.experience_replay.sample(batch_size)
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()