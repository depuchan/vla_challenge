# ACT-VLA: Action Chunking Transformer for Robotic Lift

Trained an ACT (Action Chunking Transformer) model on episodic demonstration data generated using [Robosuite](https://robosuite.ai/), simulated on the Franka Panda. The task is a standard pick-and-lift: the robot must grasp a cube from a table and raise it above a height threshold.

## Overview

The pipeline has three stages:

1. **Data collection** — A scripted expert heuristic collects 200 successful demonstration episodes and saves them to an HDF5 file.
2. **Training** — An ACT policy (ResNet-18 vision backbone + Transformer decoder) is trained with imitation learning on the collected demos. A frozen CLIP text encoder conditions the policy on the language instruction *"pick up the red cube"*.
3. **Evaluation** — The trained policy is rolled out in simulation and scored on success rate.

## Repository Structure

| File | Description |
|---|---|
| `collect_demos.py` | Scripted expert data collection → `demo_data.h5` |
| `train_act_v2.py` | Model training with AMP, torch.compile, augmentation |
| `final_eval_v2.py` | Evaluation with random cube spawns (tests generalisation) |
| `final_eval_v3.py` | Evaluation with fixed cube spawn (deterministic benchmarking) |

## Quickstart

### 1. Collect Demonstrations

```bash
python collect_demos.py
```

Change `total_episodes` inside the script to adjust dataset size. Results reported here used 200 episodes.

### 2. Train

```bash
python train_act_v2.py
```

Optimised for a T4 GPU (Google Colab). Trains for 25 epochs with a 180/20 train/val split. Saves `act_vla_model.pth` and `learning_curve.png`.

### 3. Evaluate

```bash
# Random spawn — tests generalisation
python final_eval_v2.py

# Fixed spawn — deterministic benchmark
python final_eval_v3.py
```

## Dependencies

```
robosuite
torch >= 2.0
torchvision
transformers
h5py
numpy
matplotlib
tqdm
```

## Model Architecture

- **Vision backbone**: ResNet-18 (ImageNet pretrained), global average pooled to a 512-d feature vector
- **Language conditioning**: CLIP `ViT-B/32` text encoder (frozen), CLS token projected to 512-d
- **Policy head**: 3-layer Transformer decoder with 8 attention heads, predicts action chunks of length 10
- **Action space**: 7-DoF (EEF position delta x3, orientation x3, gripper x1)
