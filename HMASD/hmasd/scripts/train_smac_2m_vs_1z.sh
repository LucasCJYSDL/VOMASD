#!/bin/sh
env="StarCraft2"
map="2m_vs_1z"
algo="hmasd"
num_env_steps=2000000
episode_length=100
skill_type="Discrete"
intri_rew_exp=0
skill_last_layer=0
skill_interval=50
team_skill_dim=3
indi_skill_dim=2
use_recurrent_discri=0
d_epoch=5
use_sparse_reward=1
lr=0.0001
policy_use_both_skill=0
lambda_env=100
lambda_team=0.1
lambda_indi=1
h_entropy_coef_start=0.005
h_entropy_coef_end=0.01
h_entropy_coef_decay=0
state_agent_specific=0
n_eval_rollout_threads=4
eval_episodes=100

seed_max=5
for seed in `seq ${seed_max}`
do
    echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed}"
    CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} \
    --num_env_steps ${num_env_steps} --episode_length ${episode_length} --skill_type ${skill_type} --skill_interval ${skill_interval} \
    --team_skill_dim ${team_skill_dim} --indi_skill_dim ${indi_skill_dim} --use_recurrent_discri ${use_recurrent_discri} \
    --d_epoch ${d_epoch} --use_sparse_reward ${use_sparse_reward} --policy_use_both_skill ${policy_use_both_skill} \
    --skill_last_layer ${skill_last_layer} --h_use_value_active_masks --l_use_value_active_masks --intri_rew_exp ${intri_rew_exp} \
    --h_lr ${lr} --h_critic_lr ${lr} --l_lr ${lr} --l_critic_lr ${lr} --d_team_lr ${lr} --d_indi_lr ${lr} \
    --lambda_team ${lambda_team} --lambda_indi ${lambda_indi} --lambda_env ${lambda_env} --state_agent_specific ${state_agent_specific} \
    --h_entropy_coef_start ${h_entropy_coef_start} --h_entropy_coef_end ${h_entropy_coef_end} --h_entropy_coef_decay ${h_entropy_coef_decay} \
    --n_eval_rollout_threads ${n_eval_rollout_threads} --eval_episodes ${eval_episodes}
done