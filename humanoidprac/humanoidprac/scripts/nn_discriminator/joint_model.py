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
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_p=0.5):
        super(JointGRUNet, self).__init__()
        gru_dropout = dropout_p if num_layers > 1 else 0.0
        self.gru = nn.GRU( 
                        input_size, 
                        hidden_size, 
                        num_layers, 
                        batch_first=True, 
                        dropout=gru_dropout  # これは中間層がある場合のみ適用
                    )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x, hidden=None):
        # 2次元入力の場合、seq_len次元を追加（ストリーミング推論用）
        # x: (batch, input_size) -> (batch, 1, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # x: (batch, sequence, input_size) when batch_first=True
        out, next_hidden = self.gru(x, hidden)
        # 全結合層に入れる前にドロップアウトを適用
        out = self.dropout(out)
        # out: (batch, sequence, hidden_size) when batch_first=True
        out = self.fc(out)  # 全バッチの最後のタイムステップを使用
        return out, next_hidden