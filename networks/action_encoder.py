import torch
import torch.nn as nn

class Action_Encoder(nn.Module): 
    def __init__(self, action_space_dim, embed_dim): 
        super().__init__()
        self.fc = nn.Linear(action_space_dim, embed_dim)
    
    def forward(self, x): 
        return self.fc(x)
    