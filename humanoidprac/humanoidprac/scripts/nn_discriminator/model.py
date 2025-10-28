import torch.nn as nn
import torch.nn.functional as F

class JointNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(JointNet, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.linear_stack(x)
        return out