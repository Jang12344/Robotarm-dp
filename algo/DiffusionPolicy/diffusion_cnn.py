#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Diffusion Policy implementation for CNN state representation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from algo.DiffusionPolicy.net_diffusion import DiffusionUNet, DiffusionMLP
from algo.DiffusionPolicy.diffusion_utils import NoiseScheduler
from config import opt
from typing import Optional, Dict, Tuple


class CNNEncoder(nn.Module):
    """CNN encoder for image observations"""
    def __init__(self, input_shape: Tuple[int, int, int], output_dim: int = 256):
        super().__init__()
        c, h, w = input_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate output dimension after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_output_dim = self.conv(dummy_input).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class DiffusionPolicy_CNN:
    """
    Diffusion Policy algorithm for vision-based robotic manipulation
    """
    def __init__(
        self,
        state_dim: Tuple[int, int, int],  # (channels, height, width)
        action_dim: int,
        action_bound: float,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        ema_decay: float = 0.995,
        clip_sample: bool = True,
        network_type: str = "unet",
        device: str = "cpu",
        horizon_steps: int = 16,
        action_horizon: int = 8
    ):
        """
        Initialize Diffusion Policy with CNN encoder
        
        Args:
            state_dim: Dimension of state space (C, H, W) for images
            action_dim: Dimension of action space
            action_bound: Bound for action values
            hidden_dim: Hidden dimension for networks
            actor_lr: Learning rate for actor network
            num_diffusion_steps: Number of diffusion steps for training
            num_inference_steps: Number of denoising steps for inference
            beta_schedule: Beta schedule type for noise scheduler
            prediction_type: Whether to predict "epsilon" (noise) or "sample" (clean action)
            ema_decay: Exponential moving average decay for target network
            clip_sample: Whether to clip predicted samples
            network_type: Type of network architecture ("unet" or "mlp")
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = torch.device(device)
        self.num_inference_steps = num_inference_steps
        self.ema_decay = ema_decay
        
        # Initialize CNN encoder
        self.cnn_encoder = CNNEncoder(state_dim, hidden_dim).to(self.device)
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_diffusion_steps=num_diffusion_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type
        )
        
        # Initialize diffusion model with encoded state dimension
        if network_type == "unet":
            self.model = DiffusionUNet(
                state_dim=hidden_dim,  # Use CNN output dimension
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                prediction_type=prediction_type
            ).to(self.device)
        else:  # mlp
            self.model = DiffusionMLP(
                state_dim=hidden_dim,  # Use CNN output dimension
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                prediction_type=prediction_type
            ).to(self.device)
            
        # EMA model for stable inference
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_cnn_encoder = copy.deepcopy(self.cnn_encoder)
        self.ema_cnn_encoder.requires_grad_(False)
        
        # Optimizer for both CNN encoder and diffusion model
        self.optimizer = torch.optim.AdamW(
            list(self.cnn_encoder.parameters()) + list(self.model.parameters()),
            lr=actor_lr,
            weight_decay=1e-6
        )
        
        # Action normalization parameters
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))
        
        self.training_steps = 0
        
    def register_buffer(self, name, tensor):
        """Register a buffer that will be part of model state"""
        setattr(self, name, tensor.to(self.device))
        
    def normalize_action(self, action):
        """Normalize action to [-1, 1]"""
        return (action - self.action_mean) / (self.action_std + 1e-8)
        
    def denormalize_action(self, action):
        """Denormalize action from [-1, 1] to original scale"""
        return action * (self.action_std + 1e-8) + self.action_mean
        
    def update_action_stats(self, actions):
        """Update action normalization statistics"""
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        self.action_mean = actions.mean(dim=0)
        self.action_std = actions.std(dim=0)
        
    @torch.no_grad()
    def update_ema(self):
        """Update EMA model parameters"""
        # Update diffusion model EMA
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        # Update CNN encoder EMA
        for param, ema_param in zip(self.cnn_encoder.parameters(), self.ema_cnn_encoder.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
            
    def take_action(self, state: np.ndarray, use_ema: bool = True) -> np.ndarray:
        """
        Generate action using diffusion process
        
        Args:
            state: Current state observation (image)
            use_ema: Whether to use EMA model for inference
            
        Returns:
            Predicted action
        """
        model = self.ema_model if use_ema else self.model
        cnn_encoder = self.ema_cnn_encoder if use_ema else self.cnn_encoder
        model.eval()
        cnn_encoder.eval()
        
        with torch.no_grad():
            # Convert state to tensor and encode
            if isinstance(state, (list, tuple)):
                state = np.array(state)
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            # Handle different input shapes for CNN
            if len(state.shape) == 3:  # (C, H, W)
                state = state.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
            elif len(state.shape) == 4:  # (1, C, H, W) or (B, C, H, W)
                pass  # Already correct
            elif len(state.shape) == 5:  # (1, 1, C, H, W) - remove extra dimension
                state = state.squeeze(1)  # -> (1, C, H, W)
            
            # Encode image state
            encoded_state = cnn_encoder(state)
            
            # Start from random noise
            noisy_action = torch.randn((1, self.action_dim), device=self.device)
            
            # Denoising loop
            for t in reversed(range(self.num_inference_steps)):
                # Create timestep tensor
                timesteps = torch.full((1,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = model(noisy_action, timesteps, encoded_state)
                
                # Denoise
                noisy_action = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=noisy_action
                )
                
            # Denormalize and clip action
            action = self.denormalize_action(noisy_action)
            action = torch.clamp(action, -self.action_bound, self.action_bound)
            
        return action.cpu().numpy()[0]
        
    def train(self, transition_dict: Dict[str, np.ndarray]) -> float:
        """
        Train diffusion model with CNN encoder
        
        Args:
            transition_dict: Dictionary containing training data
            
        Returns:
            Training loss
        """
        self.model.train()
        self.cnn_encoder.train()
        
        # Extract data
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float32).to(self.device)
        
        # Handle different input shapes for CNN states
        if len(states.shape) == 5:  # (B, 1, C, H, W) - remove extra dimension
            states = states.squeeze(1)  # -> (B, C, H, W)
        
        # Update action statistics if needed
        if self.training_steps % 1000 == 0:
            self.update_action_stats(transition_dict['actions'])
            
        # Encode states
        encoded_states = self.cnn_encoder(states)
        
        # Normalize actions
        actions = self.normalize_action(actions)
        
        # Sample random timesteps
        batch_size = actions.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.num_diffusion_steps, (batch_size,),
            device=self.device, dtype=torch.long
        )
        
        # Add noise to actions
        noise = torch.randn_like(actions)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # Predict noise
        noise_pred = self.model(noisy_actions, timesteps, encoded_states)
        
        # Calculate loss
        if self.noise_scheduler.prediction_type == "epsilon":
            target = noise
        else:  # prediction_type == "sample"
            target = actions
            
        loss = F.mse_loss(noise_pred, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.cnn_encoder.parameters()), 
            1.0
        )
        
        self.optimizer.step()
        
        # Update EMA models
        self.update_ema()
        
        self.training_steps += 1
        
        return loss.item()
        
    def save(self, filename: str):
        """Save model checkpoints"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'cnn_encoder_state_dict': self.cnn_encoder.state_dict(),
            'ema_cnn_encoder_state_dict': self.ema_cnn_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'training_steps': self.training_steps
        }, filename + "_diffusion_policy_cnn.pt")
        
    def load(self, filename: str):
        """Load model checkpoints"""
        checkpoint = torch.load(filename + "_diffusion_policy_cnn.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.cnn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
        self.ema_cnn_encoder.load_state_dict(checkpoint['ema_cnn_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']
        self.training_steps = checkpoint['training_steps'] 