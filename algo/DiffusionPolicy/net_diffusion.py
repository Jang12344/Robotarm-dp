#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Neural network architectures for Diffusion Policy
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with timestep embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.mlp(t)
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)


class DiffusionUNet(nn.Module):
    """
    U-Net architecture for diffusion models
    Predicts noise or clean samples given noisy input and timestep
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.1,
        prediction_type: str = "epsilon"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prediction_type = prediction_type
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Input projection
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dim if i == 0 else hidden_dim * (2 ** (i-1)),
                hidden_dim * (2 ** i),
                time_emb_dim,
                dropout
            )
            for i in range(num_blocks)
        ])
        
        # Middle block
        mid_dim = hidden_dim * (2 ** (num_blocks - 1))
        self.mid_block = ResidualBlock(mid_dim, mid_dim, time_emb_dim, dropout)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(
                mid_dim // (2 ** i) + (hidden_dim * (2 ** (num_blocks - 1 - i))),
                mid_dim // (2 ** (i+1)) if i < num_blocks-1 else hidden_dim,
                time_emb_dim,
                dropout
            )
            for i in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenate with state encoding
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, states: torch.Tensor):
        """
        Args:
            x: noisy actions [batch_size, action_dim]
            timesteps: diffusion timesteps [batch_size]
            states: conditional states [batch_size, state_dim]
        Returns:
            predicted noise or clean actions [batch_size, action_dim]
        """
        # Get embeddings
        t_emb = self.time_mlp(timesteps)
        s_emb = self.state_encoder(states)
        
        # Initial projection
        h = self.input_proj(x)
        
        # Encoder
        encoder_outs = []
        for block in self.encoder_blocks:
            h = block(h, t_emb)
            encoder_outs.append(h)
        
        # Middle
        h = self.mid_block(h, t_emb)
        
        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            if i < len(encoder_outs):
                h = torch.cat([h, encoder_outs[-(i+1)]], dim=-1)
            h = block(h, t_emb)
        
        # Concatenate with state encoding and project to output
        h = torch.cat([h, s_emb], dim=-1)
        out = self.output_proj(h)
        
        return out


class DiffusionMLP(nn.Module):
    """
    Simple MLP architecture for diffusion models
    Lighter weight alternative to U-Net
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        prediction_type: str = "epsilon"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prediction_type = prediction_type
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        # Main network
        layers = []
        in_dim = action_dim + state_dim + time_emb_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else action_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, states: torch.Tensor):
        """
        Args:
            x: noisy actions [batch_size, action_dim]
            timesteps: diffusion timesteps [batch_size]
            states: conditional states [batch_size, state_dim]
        Returns:
            predicted noise or clean actions [batch_size, action_dim]
        """
        t_emb = self.time_mlp(timesteps)
        h = torch.cat([x, states, t_emb], dim=-1)
        return self.net(h) 