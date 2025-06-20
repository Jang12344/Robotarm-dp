#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diffusion Policy 演示程序
展示：
1. 滚动时域优化 (Receding Horizon Control)
2. 明确的加噪和去噪过程
3. 可配置的扩散参数
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from algo.DiffusionPolicy.diffusion_mlp import DiffusionPolicy_MLP
from envs.rl_reach_env import RLReachEnv
import argparse

def demonstrate_diffusion_features():
    """演示Diffusion Policy的核心特性"""
    print("=" * 60)
    print("🤖 Diffusion Policy 增强功能演示")
    print("=" * 60)
    
    # 创建环境
    env = RLReachEnv(is_render=False, is_good_view=False)
    state_dim = 6
    action_dim = 3
    action_bound = 1.0
    
    # 创建不同配置的Diffusion Policy agents
    configs = [
        {
            "name": "快速推理配置", 
            "num_diffusion_steps": 20,
            "num_inference_steps": 5,
            "horizon_steps": 4,
            "action_horizon": 2,
            "beta_schedule": "linear"
        },
        {
            "name": "标准配置", 
            "num_diffusion_steps": 50,
            "num_inference_steps": 25,
            "horizon_steps": 8,
            "action_horizon": 4,
            "beta_schedule": "squaredcos_cap_v2"
        },
        {
            "name": "高质量配置",
            "num_diffusion_steps": 100,
            "num_inference_steps": 50,
            "horizon_steps": 16,
            "action_horizon": 8,
            "beta_schedule": "squaredcos_cap_v2"
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n📊 配置 {i+1}: {config['name']}")
        print("-" * 40)
        for key, value in config.items():
            if key != "name":
                print(f"  {key}: {value}")
        
        # 创建agent
        agent = DiffusionPolicy_MLP(
            state_dim=state_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            hidden_dim=128,
            actor_lr=3e-4,
            num_diffusion_steps=config["num_diffusion_steps"],
            num_inference_steps=config["num_inference_steps"],
            beta_schedule=config["beta_schedule"],
            prediction_type="epsilon",
            ema_decay=0.995,
            clip_sample=True,
            network_type="mlp",
            horizon_steps=config["horizon_steps"],
            action_horizon=config["action_horizon"],
            device="cpu"
        )
        
        # 演示滚动时域优化
        print(f"\n🎯 滚动时域优化演示:")
        state = env.reset()
        print(f"  初始状态: {state[:3]}...")  # 只显示前3个元素
        
        # 生成动作序列
        action_sequence = agent._generate_action_sequence(state, use_ema=False)
        print(f"  生成动作序列长度: {len(action_sequence)}")
        print(f"  预测时域: {config['horizon_steps']} 步")
        print(f"  执行时域: {config['action_horizon']} 步")
        
        # 模拟执行多步
        print(f"\n📈 执行序列:")
        for step in range(min(3, config['action_horizon'])):
            action = agent.take_action(state, replan=False)  # 不重新规划，使用序列
            print(f"  步骤 {step+1}: 动作 = [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
            # 这里可以执行 state, _, _, _ = env.step(action) 但为了演示跳过
        
        print(f"  💡 剩余序列长度: {len(agent.current_action_sequence) if agent.current_action_sequence is not None else 0}")
        
    # 演示噪声调度
    print(f"\n🔄 噪声调度对比演示:")
    print("-" * 40)
    
    schedules = ["linear", "squaredcos_cap_v2"]
    steps = 20
    
    for schedule in schedules:
        from algo.DiffusionPolicy.diffusion_utils import NoiseScheduler
        scheduler = NoiseScheduler(
            num_diffusion_steps=steps,
            beta_schedule=schedule,
            clip_sample=True,
            prediction_type="epsilon"
        )
        
        print(f"\n📉 {schedule} 调度:")
        print(f"  beta 范围: [{scheduler.betas[0]:.6f}, {scheduler.betas[-1]:.6f}]")
        print(f"  alpha 范围: [{scheduler.alphas[0]:.6f}, {scheduler.alphas[-1]:.6f}]")
        
        # 展示几个关键时间步的噪声水平
        key_steps = [0, steps//4, steps//2, 3*steps//4, steps-1]
        print(f"  关键时间步的噪声水平:")
        for t in key_steps:
            beta = scheduler.betas[t]
            alpha = scheduler.alphas[t]
            print(f"    t={t:2d}: beta={beta:.4f}, alpha={alpha:.4f}")

def demonstrate_training_process():
    """演示训练过程中的加噪和去噪"""
    print(f"\n🎓 训练过程演示:")
    print("-" * 40)
    
    # 创建简单的agent
    agent = DiffusionPolicy_MLP(
        state_dim=6, action_dim=3, action_bound=1.0,
        num_diffusion_steps=10, num_inference_steps=5,
        horizon_steps=4, action_horizon=2, device="cpu"
    )
    
    # 模拟训练数据
    states = np.random.randn(4, 6)  # 4个状态
    actions = np.random.randn(4, 3) * 0.5  # 4个动作
    
    print(f"📊 模拟训练数据:")
    print(f"  状态批次大小: {states.shape}")
    print(f"  动作批次大小: {actions.shape}")
    
    # 展示加噪过程
    print(f"\n➕ 加噪过程演示:")
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    
    # 转换为动作序列
    batch_size = actions_tensor.shape[0]
    action_sequences = actions_tensor.unsqueeze(1).repeat(1, agent.horizon_steps, 1)
    action_sequences = action_sequences.reshape(batch_size, -1)
    
    print(f"  原始动作: {actions[0]}")
    print(f"  动作序列形状: {action_sequences.shape}")
    
    # 随机时间步
    timesteps = torch.randint(0, agent.noise_scheduler.num_diffusion_steps, (batch_size,))
    print(f"  随机时间步: {timesteps.numpy()}")
    
    # 加噪
    noise = torch.randn_like(action_sequences)
    noisy_actions = agent.noise_scheduler.add_noise(action_sequences, noise, timesteps)
    
    print(f"  噪声样本: {noise[0, :3].numpy()}")  # 只显示前3个
    print(f"  加噪后动作: {noisy_actions[0, :3].numpy()}")  # 只显示前3个
    
    print(f"\n✅ 演示完成!")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Diffusion Policy 增强功能演示")
    parser.add_argument("--demo", type=str, choices=["features", "training", "all"], 
                       default="all", help="选择演示内容")
    args = parser.parse_args()
    
    if args.demo in ["features", "all"]:
        demonstrate_diffusion_features()
    
    if args.demo in ["training", "all"]:
        demonstrate_training_process()

if __name__ == "__main__":
    main() 