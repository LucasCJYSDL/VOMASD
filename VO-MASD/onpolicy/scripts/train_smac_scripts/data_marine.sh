#!/bin/sh
env="StarCraft2"
map="5m"
algo="rmappo"
exp="check"
seed=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=1 python ../../train_smac.py --data_collection --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 8000000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32 --data_collect

