# Hierarchical Multi-Agent Skill Discovery (HMASD)

This is the **official implementation** of HMASD. This repository is based on https://github.com/PKU-MARL/Multi-Agent-Transformer and https://github.com/marlbenchmark/on-policy.

## Installation

### Dependences

```shell
pip install -r requirements.txt
```

### SMAC

Please following the instructions in https://github.com/oxwhirl/smac.

### Overcooked

Please following the instructions in https://github.com/HumanCompatibleAI/overcooked_ai.

## Training

To train HMASD on Alice_and_Bob:

```shell
cd hmasd/scripts; bash train_alice_and_bob.sh
```

To train HMASD on SMAC with 0-1 reward:

```shell
# 3m
cd hmasd/scripts; bash train_smac_3m.sh
# 2s_vs_1sc
cd hmasd/scripts; bash train_smac_2s_vs_1sc.sh
# 2m_vs_1z
cd hmasd/scripts; bash train_smac_2m_vs_1z.sh
```

To train HMASD on Overcooked:

```shell
# cramped_room
cd hmasd/scripts; bash train_overcooked_cramped.sh
# asymmetric_advantages
cd hmasd/scripts; bash train_overcooked_asym.sh
# coordination_ring
cd hmasd/scripts; bash train_overcooked_coord.sh
```

Type of GPUs:

For Alice_and_Bob and SMAC with 0-1 reward, we run experiments on A40. To ensure reproducibility, we find that only one experiment can be run on one GPU card at a time. For Overcooked, we run experiments on NVIDIA GeForce RTX 3090. It allows multiple experiments on one GPU card at a time.

