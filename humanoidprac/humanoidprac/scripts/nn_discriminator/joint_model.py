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
    

class JointGRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(JointGRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x, hidden=None):
        # x: (batch, sequence, input_size) when batch_first=True
        out, next_hidden = self.gru(x, hidden)
        # out: (batch, sequence, hidden_size) when batch_first=True
        out = self.fc(out[:, -1, :])  # 全バッチの最後のタイムステップを使用
        return out, next_hidden
    