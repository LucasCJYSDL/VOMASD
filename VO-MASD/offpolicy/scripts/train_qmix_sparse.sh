#!/bin/sh
env="StarCraft2"
map="MMM2"
algo="qmix"
exp="global_alllocal"
seed=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"

echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python ../train_smac.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
    --seed ${seed} --n_training_threads 128 --buffer_size 5000 --lr 5e-4 --batch_size 32 --use_soft_update \
    --hard_update_interval_episode 200 --num_env_steps 10000000 \
    --log_interval 3000 --eval_interval 20000 --user_name "jc"\
    --use_global_all_local_state --gain 1 --use_sparse_reward
echo "training is done!"


