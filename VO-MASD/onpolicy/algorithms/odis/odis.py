import numpy as np
import torch as th
import torch.nn.functional as F
from algorithms.utils.util import check
import math
from algorithms.r_mappo.r_mappo import R_MAPPO

class ODIS(R_MAPPO):

    def __init__(self,
                 args,
                 policy,
                 device=th.device("cpu")):
        super(ODIS, self).__init__(args, policy, device)

        self.c = args.c_step
        self.skill_dim = args.skill_dim
        self.recurrent_N = args.recurrent_N
        self.hidden_size = args.hidden_size
        self.beta = args.beta
        

    def _forward_seq_action(self, obses, skill, rnn_states, avail_actions, actions, task):
        bs, n_agent = skill.shape[0], skill.shape[1]
        input_shape = bs * n_agent
        masks = th.ones(size=(input_shape, 1), dtype=th.float32, device=self.device)
        output_ls = []
        for t in range(self.c):
            action_log_probs, rnn_states = self.policy.decoder.forward(obses[:, t].reshape(input_shape, -1), rnn_states, masks, \
                                        task, skill.reshape(input_shape, -1), avail_actions[:, t].reshape(input_shape, -1),
                                        actions=actions[:, t].reshape(input_shape, -1))
            # print(action_log_probs.shape, action_log_probs)
            output_ls.append(action_log_probs.reshape(bs, n_agent, -1))
            if t == 0:
                return_rnn_states = rnn_states.clone()
        
        return th.stack(output_ls, dim=1), return_rnn_states
    
    def pretrain(self, batch, task):
        states, obses, actions, avail_actions, masks = batch["state"], batch["obs"], batch["action"], batch["avail_action"], batch["filled"]
        states, obses, avail_actions, masks = check(states).to(**self.tpdv), check(obses).to(**self.tpdv),\
                                              check(avail_actions).to(**self.tpdv), check(masks).to(**self.tpdv)
        actions = check(actions).to(**dict(dtype=th.long, device=self.device))
        bs, seq_len, n_agent = states.shape[0], states.shape[1], states.shape[2]

        # compute skill logits, this is not an rnn process
        skill_logits = self.policy.state_encoder(states.reshape(bs * seq_len * n_agent, -1), actions.reshape(bs * seq_len * n_agent, -1), task)
        skill_logits = skill_logits.reshape(bs, seq_len, n_agent, -1)

        ######## beta-vae loss
        # prior loss
        seq_skill_input = F.gumbel_softmax(skill_logits[:, :-self.c, :, :], dim=-1)
        # print("1: ", seq_skill_input.shape, seq_skill_input[0], mask[:, :-self.c].shape) # torch.Size([32, 25, 3, 5]), probability distribution
        kl_seq_skill = ((seq_skill_input * (th.log(seq_skill_input) - math.log(1 / self.skill_dim))) * masks[:, :-self.c]).sum() / masks[:, :-self.c].sum()
        enc_loss = kl_seq_skill

        # print("0:", kl_seq_skill)
        dec_loss = 0.   ### batch time agent skill
        
        rnn_states = np.zeros((bs * n_agent, self.recurrent_N, self.hidden_size), dtype=np.float32)
        fwd_times = 0
        for t in range(seq_len-self.c): 
            action_log_probs, rnn_states = self._forward_seq_action(obses[:, t:t+self.c], seq_skill_input[:, t, :, :], \
                                                                    rnn_states, avail_actions[:, t:t+self.c], actions[:, t:t+self.c], task)
            # print(b, c, n, a) # 32 2 3 21
            if masks[:, t:t+self.c].sum().item() == 0:
                break
            fwd_times += 1
            dec_loss += - (action_log_probs.reshape(-1) * masks[:, t:t+self.c].reshape(-1)).sum() / masks[:, t:t+self.c].sum() # TODO: cross entropy
            # print("1: ", dec_loss, masks[:, t:t+self.c].sum())
            
        vae_loss = dec_loss / float(fwd_times) + self.beta * enc_loss
        loss = vae_loss

        loss.backward()
        # for name, parameter in self.policy.named_parameters(): 
        #     print(name, parameter.grad.sum() if parameter.grad is not None else None)

        return {"enc_loss": enc_loss.item(), "dec_loss": dec_loss.item(), "tot_loss": loss.item()}

    def pretrain_update(self):
        grad_norm = th.nn.utils.clip_grad_norm_(self.policy.pretrain_parameters, self.max_grad_norm)
        
        self.policy.pretrain_optimizer.step()
        self.policy.pretrain_optimizer.zero_grad()
    

