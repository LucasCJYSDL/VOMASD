# Variational Offline Multi-agent Skill Discovery

## Environment:
- Please follow the "Installation instructions" section in [https://github.com/LAMDA-RL/ODIS/blob/main/README.md](https://github.com/LAMDA-RL/ODIS/blob/main/README.md).

## Data
- You can either download the multi-task offline data from [Google Drive](https://drive.google.com/drive/folders/1tbYHapsxTSRmjG1h-KOhdsYrQNby-pCC?usp=drive_link), or generate new data with our provided checkpoints:
```bash
cd VO-MASD/onpolicy/scripts/train_smac_scripts
./data_XXX.sh
```

- Here, XXX can be one of [marine, MMM, MMM2, terran] for running marine, MMM, MMM2, terran tasks, repectively. Marine tasks include 3m and 5m, so please change the map name in data_marine.sh (i.e., Line 3) accordingly. If you want to generate mediate-level data, please use the '--is_med' command by adding it in Line 13 of `data_XXX.sh'.

- The datasets need to be put under the folder 'data'.

## Evaluation on the utility of the discovered skills
- We provide comparisons among [vomasd-3d, vomasd-single, vomasd-hier, vomasd-mixed, odis] on [3m, 5m, 7m, 10m, MMM, MMM2], which can be reproduced by:

```bash
cd VO-MASD/onpolicy/scripts/train_smac_scripts
./train_XXX.sh
```

- XXX can be one of [marine, MMM, MMM2]. Marine tasks include [3m, 5m, 7m, 10m], so please change the map name in train_marine.sh (i.e., Line 3) accordingly.

- To change the algorithm for evaluation, please modify Line 4 of train_XXX.sh to one of [vomasd-3d, vomasd-single, vomasd-hier, vomasd-mixed, odis].

- To change the random seed, please modify Line 6 of train_XXX.sh, for which we choose from [1, 2, 3].

- To reproduce results on mediate-level or mixed-level data, please add the '--is_med' or '--is_mixed' command on Line 13 of 'train_XXX.sh', respectively. 

- A special hyperparamter is the 'pretrain_steps', which we set as 1000 or 2000 to balance the performance and training cost.

- Comparisons among [vomasd-3d, vomasd-hier, odis] on [terran_3, terran_5, terran_7] can be reproduced by:

```bash
cd VO-MASD/onpolicy/scripts/train_smac_scripts
./train_terran.sh
```
- The options for tasks, algorithms, and random seeds can be set in the same way as mentioned above.


## Evaluation on the sparse-reward setup
- We provide comparisons among [vomasd-3d, vomasd-hier, vomasd-mixed, rmappo, qmix] on unseen tasks [7m, 10m, MMM2], which can be reproduced by (for [vomasd-3d, vomasd-hier, vomasd-mixed]):

```bash
cd VO-MASD/onpolicy/scripts/train_smac_scripts
./train_XXX_sparse.sh
```

- XXX can be one of [marine, MMM2]. Marine tasks include [7m, 10m], so please change the map name in train_marine_sparse.sh (i.e., Line 3) accordingly.

- To change the algorithm for evaluation, please modify Line 4 of train_XXX_sparse.sh to one of [vomasd-3d, vomasd-hier, vomasd-mixed, rmappo].

- For qmix, please run: ('map' in Line 3 can be one of [7m, 10m, MMM2])

```bash
cd offpolicy/scripts
./train_qmix_sparse.sh
```

## Evaluation results of HMASD

- For the results on the sparse-reward setup:
```bash
cd HMASD/hmasd/scripts/train
./train_smac_XXX.sh
```
- XXX can be one of [7m, 10m, MMM2].

- For the results on terran:
```bash
cd HMASD/hmasd/scripts/train
./train_terran.sh
```

- Please change its Line 3 into one of [terran_3, terran_5, terran_7] to run corresponding experiments.


## Reference

- [ODIS](https://github.com/LAMDA-RL/ODIS)
- [MAPPO](https://github.com/marlbenchmark)
- [HMASD](https://openreview.net/forum?id=xMgO04HDOS)