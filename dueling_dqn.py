import torch.nn as nn

class DuelingDQN(nn.Module):

    def __init__(self, input_size=16, output_size=4):
        super(DuelingDQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU()
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.output_size)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value(features)
        advantages = self.advantage(features)
        q_values = values + (advantages - advantages.mean())
        
        return q_values