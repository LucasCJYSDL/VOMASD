import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from hmasd.algorithms.utils.util import init, check
from hmasd.algorithms.utils.cnn import CNNBase
from hmasd.algorithms.utils.mlp import MLPBase
from hmasd.algorithms.utils.rnn import RNNLayer
from hmasd.algorithms.utils.discri import DiscriLayer

class R_Discri(nn.Module):

    def __init__(self, args, input_dim, output_dim, output_type, device=torch.device("cpu")):
        super(R_Discri, self).__init__()
        self.intri_rew_exp= args.intri_rew_exp
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_discri
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.output_type = output_type

        base = MLPBase
        self.base = base(args, input_dim)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.output = DiscriLayer(self.hidden_size, output_dim, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, input, rnn_states, masks):
        # roll_out
        # input: (batch, input_dim)
        # rnn_states: (batch, recurrent_N, hidden_size)
        # masks: (batch, 1)
        # train
        # input: (data_chunk_length*mini_batch_size, state_dim)
        # rnn_states: (mini_batch_size, recurrent_N, hidden_size)
        # masks: (data_chunk_length*mini_batch_size, 1)
        input = check(input).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        x = self.base(input)

        if self._use_recurrent_policy:
            x, rnn_states = self.rnn(x, rnn_states, masks)

        logits = self.output(x)

        return logits, rnn_states
        # roll_out (batch, output_dim), (batch, recurrent_N, hidden_size)
        # train (data_chunk_length*mini_batch_size, output_dim), (mini_batch_size, recurrent_N, hidden_size)

    def get_intrinsic_reward(self, input, rnn_states, skill, masks):
        # input: (batch, input_dim)
        # rnn_states: (batch, recurrent_N, hidden_size)
        # skill: (batch, skill_num)
        # masks: (batch, 1)
        logits, rnn_states = self.forward(input, rnn_states, masks) # (batch, output_dim)
        skill = check(skill).to(**self.tpdv)
        if self.output_type == 'Discrete':
            log_prob = log_softmax(logits, dim=-1)
            intri_rew = torch.gather(log_prob, -1, skill.long()) # (batch, 1)
        else:
            intri_rew = -((logits - skill) ** 2).mean(-1, keepdim=True) # (batch, 1)
        if self.intri_rew_exp:
            intri_rew = torch.exp(intri_rew)

        return intri_rew, rnn_states # (batch, 1), (batch, recurrent_N, hidden_size)
