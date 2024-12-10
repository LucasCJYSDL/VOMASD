import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def binary_embed(v, length, v_max):
    assert 2**length - 1 >= v_max
    embed_vec = np.zeros(length,)
    bin_v = [int(item) for item in list(bin(v)[2:])]
    embed_vec[-len(bin_v):] = bin_v
    return embed_vec

class Item:
    def __init__(self, group, code, distance):
        self.group = group
        self.code = code
        self.distance = distance
        
    def __lt__(self, other):
        return self.distance < other.distance

    def __repr__(self):
        return f"{self.group} and {self.code} have a distance of ({self.distance})."