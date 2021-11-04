
import random
import torch
from collections import deque

class ExperienceReply:

    def __init__(self, max_size):
        self.max_size = max_size
        self.replay = deque(maxlen=max_size)
        
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.replay.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.replay, batch_size)
        state_batch = torch.cat([s1 for (s1, _, _, _, _) in batch]) 
        action_batch = [a for (_, a, _, _, _) in batch]
        reward_batch = [r for (_, _, r, _, _) in batch]
        next_state_batch = torch.cat([s2 for (_, _, _, s2, _) in batch]) 
        done_batch = [d for (_, _, _, _, d) in batch]
        
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
    
    def __len__(self):
        return len(self.replay)