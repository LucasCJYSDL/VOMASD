import torch.nn as nn
from .util import init

class DiscriLayer(nn.Module):

    def __init__(self, input_dim, output_dim, use_orthogonal, gain):
        super(DiscriLayer, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(input_dim, output_dim))
    
    def forward(self, x):
        x = self.linear(x)
        return x