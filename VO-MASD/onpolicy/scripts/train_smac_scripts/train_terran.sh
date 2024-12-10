#!/bin/sh
env="StarCraft2_v2"
map="terran_3"
algo="vomasd-hier"
exp="check"
seed=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 8000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 --pretrain_steps 1000 --c_step 5
