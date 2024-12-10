import numpy as np
import torch as th
import torch.nn.functional as F
from algorithms.utils.util import check
from algorithms.r_mappo.r_mappo import R_MAPPO
from utils.valuenorm import ValueNorm
from utils.util import pre_feed_forward_generator, huber_loss, mse_loss



def _t2n(x):
    return x.detach().cpu().numpy()


class VOMASD3d(R_MAPPO):

    def __init__(self,
                 args,
                 policy,
                 device=th.device("cpu")):
        super(VOMASD3d, self).__init__(args, policy, device)

        self.c = args.c_step
        self.skill_dim = args.skill_dim
        self.skill_num = args.skill_num

        self.recurrent_N = args.recurrent_N
        self.hidden_size = args.hidden_size
        self.beta = args.beta

        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm

        if self._use_popart:
            self.pre_value_normalizer = self.policy.grouper.v_out
        elif self._use_valuenorm:
            self.pre_value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.pre_value_normalizer = None


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

    def _forward_vae(self, batch, task, vae_training):
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
        
        z_q, code_diff, group, log_group, v_out, emb_rwd = self.policy.forward_code(states[:, :proc_seq_len], z_e, task, test_mode=vae_training)
        # torch.Size([32, 25, 3, 10]) torch.Size([800, 3, 1]) torch.Size([800, 3, 1]) torch.Size([800, 3, 1]) torch.Size([800, 3, 1])
        emb_loss = (code_diff * masks[:, :proc_seq_len].reshape(-1, n_agent, 1)).sum() / masks[:, :proc_seq_len].sum()

        dec_loss = 0.0   
        # TODO: put the rnn_states initialization inside the loop
        rnn_states = np.zeros((bs * n_agent, self.recurrent_N, self.hidden_size), dtype=np.float32)
        fwd_times = 0
        rec_rwd = check(th.zeros((bs, proc_seq_len, n_agent, 1))).to(**self.tpdv)
        for t in range(seq_len-self.c): 
            action_log_probs, rnn_states = self._forward_seq_action(obses[:, t:t+self.c], z_q[:, t, :, :], \
                                                                    rnn_states, avail_actions[:, t:t+self.c], actions[:, t:t+self.c], task)
            # print(b, c, n, a) # 32 2 3 21
            if masks[:, t:t+self.c].sum().item() == 0:
                break
            fwd_times += 1
            dec_loss += - (action_log_probs.reshape(-1) * masks[:, t:t+self.c].reshape(-1)).sum() / masks[:, t:t+self.c].sum() # TODO: cross entropy
            rec_rwd[:, t] = (action_log_probs * masks[:, t:t+self.c]).sum(dim=1).detach().clone() # ([32, 3, 1])
        
        rec_rwd = rec_rwd.reshape(-1, n_agent, 1)
        vae_loss = dec_loss / float(fwd_times) 
        loss = vae_loss + emb_loss # TODO: add a weight here, note that if only using vae_loss, there is no gradients for codebooks

        if vae_training:
            return loss, {"tot_loss": loss, "vae_loss": vae_loss, "emb_loss": emb_loss}
        
        return _t2n(states[:, :proc_seq_len].reshape(-1, n_agent, states.shape[-1])), _t2n(group),\
               _t2n(log_group), _t2n(rec_rwd + emb_rwd), _t2n(v_out.reshape(-1, n_agent, 1)),\
               _t2n(masks[:, :proc_seq_len].reshape(-1, n_agent, 1))  # TODO: add a weight parameter on the return

    def pretrain(self, batch, task):
        loss, vae_train_info = self._forward_vae(batch, task, vae_training=True)
        loss.backward()
        # for name, parameter in self.policy.named_parameters(): 
        #     print(name, parameter.grad.sum() if parameter.grad is not None else None)
        # raise NotImplementedError

        return vae_train_info
    
    def pretrain_rl(self, batch, task):
        states, groups, log_groups, returns, values, masks = self._forward_vae(batch, task, vae_training=False)
        
        if self._use_popart or self._use_valuenorm:
            advantages = returns - self.pre_value_normalizer.denormalize(values)
        else:
            advantages = returns - values

        advantages_copy = advantages.copy()
        advantages_copy[masks == 0.0] = np.nan 
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = pre_feed_forward_generator(states, groups, log_groups, advantages, values, \
                                                        returns, masks, num_mini_batch=self.num_mini_batch)

            for sample in data_generator:
                value_loss, policy_loss, dist_entropy = self.pre_ppo_update(sample, task)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info
    
    def pre_cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.pre_value_normalizer.update(return_batch.reshape(-1, 1))
            error_clipped = self.pre_value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.pre_value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = th.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum() # torch.Size([3200, 3, 1]) torch.Size([3200, 3, 1])

        return value_loss
    
    def pre_ppo_update(self, sample, task):

        share_obs_batch, actions_batch, value_preds_batch, return_batch,\
              masks_batch, old_action_log_probs_batch, adv_targ = sample
        # print("here: ", available_actions_batch)

        share_obs_batch = check(share_obs_batch).to(**self.tpdv) # torch.Size([3200, 3, 117])
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)

        actions_batch = check(actions_batch).to(**dict(dtype=th.long, device=self.device))

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.grouper.evaluate_actions(share_obs_batch, actions_batch, task)
        # actor update
        imp_weights = th.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = th.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_action_loss = (-th.sum(th.min(surr1, surr2), dim=-1, keepdim=True) * masks_batch).sum() / masks_batch.sum()
        dist_entropy = (dist_entropy * masks_batch).sum() / masks_batch.sum()

        policy_loss = policy_action_loss - dist_entropy * self.entropy_coef

        # critic update
        value_loss = self.pre_cal_value_loss(values, value_preds_batch, return_batch, masks_batch)

        value_loss = value_loss * self.value_loss_coef

        (policy_loss + value_loss).backward()
        # for name, parameter in self.policy.named_parameters(): 
        #     print(name, parameter.grad.sum() if parameter.grad is not None else None)
        # raise NotImplementedError

        self.pretrain_update()

        return value_loss, policy_loss, dist_entropy
        

    def pretrain_update(self):
        grad_norm = th.nn.utils.clip_grad_norm_(self.policy.pretrain_parameters, self.max_grad_norm)
        
        self.policy.pretrain_optimizer.step()
        self.policy.pretrain_optimizer.zero_grad()
    
    
    
