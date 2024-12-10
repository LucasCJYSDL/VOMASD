#!/usr/bin/env python
import sys
import os
# import wandb
# import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config

from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

"""Train script for SMAC."""

def make_env(all_args, is_eval):
    if all_args.env_name == "StarCraft2_v2":
        from envs.starcraft2_v2.StarCraft2_v2_Env import StarCraft2Env
        from envs.starcraft2_v2.wrapper import StarCraftCapabilityEnvWrapper
    else:
        from envs.starcraft2.StarCraft2_Env import StarCraft2Env

    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            elif all_args.env_name == "StarCraft2_v2":
                agent_num = int(all_args.map_name.split('_')[-1])
                distribution_config = {
                                        "n_units": agent_num,
                                        "n_enemies": agent_num,
                                        "team_gen": {
                                            "dist_type": "weighted_teams",
                                            "unit_types": ["marine", "marauder", "medivac"],
                                            "exception_unit_types": ["medivac"],
                                            "weights": [0.45, 0.45, 0.1],
                                            "observe": True,
                                        },
                                        "start_positions": {
                                            "dist_type": "surrounded_and_reflect",
                                            "p": 0.5,
                                            "n_enemies": agent_num,
                                            "map_x": 32,
                                            "map_y": 32,
                                        }
                                    }
                env = StarCraftCapabilityEnvWrapper(
                    capability_config=distribution_config,
                    map_name="10gen_terran",
                    debug=False,
                    conic_fov=False,
                    obs_own_pos=True,
                    use_unit_ranges=True,
                    min_attack_range=2,
                    max_agent_id=all_args.max_agent_id)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env
    
    if is_eval:
        n_threads = all_args.n_eval_rollout_threads
    else:
        n_threads = all_args.n_rollout_threads

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


# def make_eval_env(all_args):
#     if all_args.env_name == "StarCraft2_v2":
#         from envs.starcraft2_v2.StarCraft2_v2_Env import StarCraft2Env
#     else:
#         from envs.starcraft2.StarCraft2_Env import StarCraft2Env

#     def get_env_fn(rank):
#         def init_env():
#             if all_args.env_name == "StarCraft2":
#                 env = StarCraft2Env(all_args)
#             else:
#                 print("Can not support the " + all_args.env_name + "environment.")
#                 raise NotImplementedError
#             env.seed(all_args.seed * 50000 + rank * 10000)
#             return env

#         return init_env

#     if all_args.n_eval_rollout_threads == 1:
#         return ShareDummyVecEnv([get_env_fn(0)])
#     else:
#         return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_env(all_args, is_eval=False)
    eval_envs = make_env(all_args, is_eval=True) if all_args.use_eval else None

    if all_args.env_name == "StarCraft2_v2":
        num_agents = int(all_args.map_name.split('_')[-1])
    else:
        from envs.starcraft2.smac_maps import get_map_params
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.algorithm_name == "rmappo":
        from runner.shared.smac_runner import SMACRunner as Runner
        runner = Runner(config)
    else:
        if all_args.algorithm_name in ["odis", "vomasd-single"]:
            from runner.shared.hier_runner import HierRunner as Runner
        else:
            from runner.shared.vomasd_runner import VOMASDRunner as Runner
        data_path = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/data")
        runner = Runner(config, data_path)

    if all_args.data_collection:
        path_str = all_args.map_name
        if all_args.is_med:
            path_str = all_args.map_name + '-med'
        if all_args.is_random:
            path_str = all_args.map_name + '-random'
        load_path = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/ckpt") / path_str / "models"
        down_path = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/data") / path_str
        if not down_path.exists():
            os.makedirs(str(down_path))
        # print(load_path, down_path)
        episode_number = 200 // all_args.n_rollout_threads # 100 for MMM, 32 for MMM2
        runner.collect_data(load_path, down_path, episode_number)
    else:
        runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
