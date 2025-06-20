# Diffusion Policy pybullet Usage Guide:
Author Contact: reicholguk@proton.me
## 🎯 Overview

Diffusion Policy is a powerful behavior cloning method that models action sequences using diffusion models. This implementation includes receding horizon optimization and supports both MLP and U-Net architectures.This directory contains the implementation of Diffusion Policy for robotic manipulation tasks.The basic envs for the robotarm on pybullet by reinforcement learning refers to https://github.com/Shimly-2/DRL-on-robot-arm.git.


### Key Features
- **Receding Horizon Optimization**: Predicts action sequences and executes them step by step
- **Flexible Network Architecture**: Supports both MLP and U-Net architectures  
- **Multi-modal Inputs**: Supports both vector states and image observations
- **HER Integration**: Compatible with Hindsight Experience Replay
- **Detailed Logging**: Comprehensive denoising process visualization

## 🚀 Quick Start

### Vector State Input (MLP)
```bash
python main.py train_reach_with_DiffusionPolicy \
    --num_diffusion_steps=100 \
    --num_inference_steps=50 \
    --diffusion_lr=0.0003 \
    --horizon_steps=16 \
    --action_horizon=8
```

### Image Input (CNN)
```bash
python main.py train_reach_with_DiffusionPolicy_CNN \
    --network_type=unet \
    --prediction_type=epsilon \
    --beta_schedule=squaredcos_cap_v2 \
    --horizon_steps=16 \
    --action_horizon=8
```

## ⚙️ Parameter Configuration

### Core Diffusion Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_diffusion_steps` | 100 | Number of forward diffusion steps (noise addition) |
| `num_inference_steps` | 50 | Number of reverse inference steps (denoising) |
| `diffusion_lr` | 0.0003 | Learning rate for diffusion model |
| `beta_schedule` | "squaredcos_cap_v2" | Noise schedule: "linear" or "squaredcos_cap_v2" |
| `prediction_type` | "epsilon" | Prediction target: "epsilon" (noise) or "sample" (clean action) |
| `ema_decay` | 0.995 | Exponential moving average decay for model parameters |

### Receding Horizon Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon_steps` | 16 | Prediction horizon length, how many steps to predict at once |
| `action_horizon` | 8 | Number of actions to actually execute, typically half of horizon_steps |

### Network Architecture Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_type` | "mlp" | Network type: "mlp" or "unet" |
| `clip_sample` | True | Whether to clip predicted samples |

## Usage Examples

### Example 1: Fast Inference Configuration
```bash
# Reduce inference steps for faster execution
python main.py train_reach_with_DiffusionPolicy \
    --num_diffusion_steps=50 \
    --num_inference_steps=20 \
    --horizon_steps=8 \
    --action_horizon=4
```

### Example 2: High-Quality Configuration
```bash
# Increase steps for better quality
python main.py train_reach_with_DiffusionPolicy \
    --num_diffusion_steps=200 \
    --num_inference_steps=100 \
    --horizon_steps=32 \
    --action_horizon=16
```

### Example 3: Image-based Task
```bash
# CNN version for camera observations
python main.py train_reach_with_DiffusionPolicy_CNN \
    --network_type=unet \
    --num_diffusion_steps=100 \
    --num_inference_steps=50 \
    --diffusion_lr=0.0001
```

## 🔧 Advanced Features

### 1. Receding Horizon Optimization
- **Principle**: Predict action sequences of length `horizon_steps`, but only execute the first `action_horizon` actions
- **Advantage**: Provides long-term planning while maintaining real-time execution
- **Configuration**: Typically set `action_horizon = horizon_steps // 2`

### 2. Multi-scale Network Architecture
- **MLP**: Suitable for low-dimensional vector inputs, fast training
- **U-Net**: Suitable for high-dimensional image inputs, better feature extraction

### 3. Flexible Noise Schedules
- **Linear**: Simple linear noise increase, suitable for quick experiments
- **Cosine**: More gradual noise increase, typically yields better results

### 4. Prediction Types
- **Epsilon**: Predict noise, more stable training
- **Sample**: Predict clean actions directly, more intuitive

## 📊 Performance Monitoring

The system automatically logs the following metrics:
- `diffusion_loss`: Training loss
- `avg_return`: Average episode return
- `success_rate`: Task success rate
- Detailed denoising process logs

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce `horizon_steps` and `action_horizon`
   - Use smaller batch sizes

2. **Training Instability**
   - Try `prediction_type="epsilon"`
   - Reduce learning rate
   - Increase `ema_decay`

3. **Slow Inference**
   - Reduce `num_inference_steps`
   - Use `network_type="mlp"` for simple tasks

## 📈 Performance Optimization

### Training Speed
- Use MLP for vector inputs
- Reduce diffusion steps during development
- Use smaller horizons for simple tasks

### Training Quality
- Use U-Net for image inputs
- Increase diffusion and inference steps
- Use cosine noise schedule
- Enable EMA for stable inference

## 🎯 Best Practices

1. **Parameter Selection**
   - Start with default parameters
   - Gradually adjust based on task complexity
   - Monitor success rate and convergence

2. **Network Choice**
   - MLP: For vector states (positions, velocities)
   - U-Net: For image observations

3. **Horizon Setting**
   - Longer horizons: Better for planning tasks
   - Shorter horizons: Better for reactive tasks

4. **Training Strategy**
   - Use HER for sparse reward tasks
   - Monitor both loss and success rate
   - Save models with high success rates

## 🎭 Demo Program

Run demo program to see all features:

```bash
# Full demo
python demo_diffusion.py

# Only demo feature capabilities
python demo_diffusion.py --demo features

# Only demo training process
python demo_diffusion.py --demo training
```

## 🐛 Common Questions

### Q1: Action output unstable?
**A**: Try increasing `num_inference_steps` or using larger `ema_decay`

### Q2: Training slow to converge?
**A**: Adjust learning rate `diffusion_lr`, or try "sample" prediction type

### Q3: Memory issues?
**A**: Reduce `batch_size`, `horizon_steps`, or `hidden_dim`

### Q4: CNN version dimension error?
**A**: Check input image dimensions, ensure (C, H, W) format

## 📈 Experiment Suggestions

1. **Start with quick configuration and validate**
2. **Gradually increase complexity**
3. **Compare different noise schedules**
4. **Adjust horizon parameters to observe effects**
5. **Monitor training loss and success rate**

## 🔗 Related Resources

- [Diffusion Policy Paper](https://arxiv.org/abs/2303.04137)
- [Original Implementation](https://github.com/columbia-ai-robotics/diffusion_policy)


