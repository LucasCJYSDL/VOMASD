import numpy as np
import torch as th
import torch.nn.functional as F
from algorithms.utils.util import check
from algorithms.r_mappo.r_mappo import R_MAPPO

class VOMASDSingle(R_MAPPO):

    def __init__(self,
                 args,
                 policy,
                 device=th.device("cpu")):
        super(VOMASDSingle, self).__init__(args, policy, device)

        self.c = args.c_step
        self.skill_dim = args.skill_dim
        self.skill_num = args.skill_num
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

        # begin the pretrain process

        ## get the skill embedding
        obs_input, act_input = [], []
        for t in range(seq_len - self.c):
            obs_input.append(obses[:, t:t+self.c])
            act_input.append(actions[:, t:t+self.c])
        obs_input = th.stack(obs_input, dim=1)
        act_input = th.stack(act_input, dim=1)

        proc_seq_len = obs_input.shape[1]
        skill_embs = self.policy.traj_encoder(obs_input.reshape(bs * proc_seq_len, self.c, n_agent, -1),\
                                              act_input.reshape(bs * proc_seq_len, self.c, n_agent, -1), task)

        z_e = skill_embs.reshape(bs, proc_seq_len, n_agent, self.skill_dim)
        z_q = self.policy.forward_code(z_e)
        # print("0:", proc_seq_len, z_e.shape, z_q.shape, skill_embs.shape) # 25 torch.Size([32, 25, 3, 5]) torch.Size([32, 25, 3, 5]) torch.Size([2400, 5])
        # TODO: adjust the parameter beta
        emb_loss = ((th.mean((z_q.detach() - z_e) ** 2, dim=-1, keepdim=True) + \
                     self.beta * th.mean((z_q - z_e.detach()) ** 2, dim=-1, keepdim=True)) * masks[:, :proc_seq_len]).sum() / masks[:, :proc_seq_len].sum()
        
        z_q = z_e + (z_q - z_e).detach()
        
        dec_loss = 0.0   
        # TODO: put the rnn_states initialization inside the loop
        rnn_states = np.zeros((bs * n_agent, self.recurrent_N, self.hidden_size), dtype=np.float32)
        fwd_times = 0
        for t in range(seq_len-self.c): 
            action_log_probs, rnn_states = self._forward_seq_action(obses[:, t:t+self.c], z_q[:, t, :, :], \
                                                                    rnn_states, avail_actions[:, t:t+self.c], actions[:, t:t+self.c], task)
            # print(b, c, n, a) # 32 2 3 21
            if masks[:, t:t+self.c].sum().item() == 0:
                break
            fwd_times += 1
            dec_loss += - (action_log_probs.reshape(-1) * masks[:, t:t+self.c].reshape(-1)).sum() / masks[:, t:t+self.c].sum() # TODO: cross entropy

        
        vae_loss = dec_loss / float(fwd_times) 

        loss = vae_loss + emb_loss # TODO: add a weight here, note that if only using vae_loss, there is no gradients for codebooks

        loss.backward()
        # for name, parameter in self.policy.codebook.named_parameters(): 
        #     print(name, parameter.grad.sum() if parameter.grad is not None else None)
        # raise NotImplementedError

        return {"tot_loss": loss, "vae_loss": vae_loss, "emb_loss": emb_loss}

    def pretrain_update(self):
        grad_norm = th.nn.utils.clip_grad_norm_(self.policy.pretrain_parameters, self.max_grad_norm)
        
        self.policy.pretrain_optimizer.step()
        self.policy.pretrain_optimizer.zero_grad()
    
    
    
