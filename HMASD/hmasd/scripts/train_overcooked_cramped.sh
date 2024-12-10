#!/bin/sh
env="overcooked"
map="cramped_room"
algo="hmasd"
num_env_steps=10000000
episode_length=400
skill_type="Discrete"
intri_rew_exp=0
skill_last_layer=1
skill_interval=25
team_skill_dim=3
indi_skill_dim=3
use_recurrent_discri=0
d_epoch=15
lr=0.0001
policy_use_both_skill=0
lambda_env=100
lambda_team=1
lambda_indi=0.5
h_entropy_coef_start=0.01
h_entropy_coef_end=0.01
h_entropy_coef_decay=0
n_eval_rollout_threads=8

seed_max=5
for seed in `seq ${seed_max}`
do
    echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed}"
    CUDA_VISIBLE_DEVICES=0 python train/train_overcooked.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} \
    --num_env_steps ${num_env_steps} --episode_length ${episode_length} --skill_type ${skill_type} --skill_interval ${skill_interval} \
    --team_skill_dim ${team_skill_dim} --indi_skill_dim ${indi_skill_dim} --use_recurrent_discri ${use_recurrent_discri} \
    --d_epoch ${d_epoch} --policy_use_both_skill ${policy_use_both_skill} --n_eval_rollout_threads ${n_eval_rollout_threads} \
    --skill_last_layer ${skill_last_layer} --intri_rew_exp ${intri_rew_exp} \
    --h_lr ${lr} --h_critic_lr ${lr} --l_lr ${lr} --l_critic_lr ${lr} --d_team_lr ${lr} --d_indi_lr ${lr} \
    --lambda_team ${lambda_team} --lambda_indi ${lambda_indi} --lambda_env ${lambda_env} \
    --h_entropy_coef_start ${h_entropy_coef_start} --h_entropy_coef_end ${h_entropy_coef_end} --h_entropy_coef_decay ${h_entropy_coef_decay}
done