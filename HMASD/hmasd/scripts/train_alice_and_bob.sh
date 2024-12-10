#!/bin/sh
env="alice_and_bob"
algo="hmasd"
game_version=0
num_env_steps=3000000
episode_length=100
skill_type="Discrete"
intri_rew_exp=0
skill_last_layer=1
skill_interval=50
team_skill_dim=2
indi_skill_dim=4
use_recurrent_discri=0
d_epoch=15
lr=0.0005
policy_use_both_skill=0
h_entropy_coef=0.1
lambda_env=0.0
lambda_team=0.1
lambda_indi=0.2
eval_episodes=100

seed_max=5
for seed in `seq ${seed_max}`
do
    echo "env is ${env}, algo is ${algo}, seed is ${seed}"
    CUDA_VISIBLE_DEVICES=0 python train/train_alice_and_bob.py --game_version ${game_version} --env_name ${env} --algorithm_name ${algo} --seed ${seed} \
    --num_env_steps ${num_env_steps} --episode_length ${episode_length} --skill_type ${skill_type} --skill_interval ${skill_interval} \
    --team_skill_dim ${team_skill_dim} --indi_skill_dim ${indi_skill_dim} --use_recurrent_discri ${use_recurrent_discri} \
    --d_epoch ${d_epoch} --skill_last_layer ${skill_last_layer} --intri_rew_exp ${intri_rew_exp} --h_entropy_coef ${h_entropy_coef} \
    --h_lr ${lr} --h_critic_lr ${lr} --l_lr ${lr} --l_critic_lr ${lr} --d_team_lr ${lr} --d_indi_lr ${lr} --policy_use_both_skill ${policy_use_both_skill} \
    --lambda_team ${lambda_team} --lambda_indi ${lambda_indi} --lambda_env ${lambda_env} --eval_episodes ${eval_episodes}
done
