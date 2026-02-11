import torch
import torch.nn as nn

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.liner = nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.liner2 = nn.Linear(ffn_num_hiddens,ffn_num_outputs)

    def forward(self,X):
        return self.liner2(self.relu(self.liner(X)))