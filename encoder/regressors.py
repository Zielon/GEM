# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2025 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: wojciech.zielonka@tuebingen.mpg.de, wojciech.zielonka@tu-darmstadt.de


import torch.nn as nn
import torch
import math


class MLP(nn.Module):
    def __init__(self, input_dim, n_parts=10, outsize=70):
        super().__init__()
        self.num_parts = n_parts
        hdim = 256

        self.parts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.2),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, outsize)
            )
            for _ in range(self.num_parts)
        ])

    def forward(self, x):
        outputs = []
        for mlp in self.parts:
            part_out = mlp(x)
            outputs.append(part_out)

        coeffs = torch.cat(outputs, dim=-1)
        return coeffs


class AttentionMLP(nn.Module):
    def __init__(self, input_dim, n_parts=10, outsize=70):
        super().__init__()
        self.num_parts = n_parts
        hdim = 256

        self.attn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            )
            for _ in range(self.num_parts)
        ])

        self.parts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.2),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, outsize)
            )
            for _ in range(self.num_parts)
        ])

    def forward(self, x):
        outputs = []

        for attn_layer, mlp in zip(self.attn_layers, self.parts):
            weights = attn_layer(x)
            weighted_x = x * weights
            part_out = mlp(weighted_x)
            outputs.append(part_out)

        coeffs = torch.cat(outputs, dim=-1)
        return coeffs


class GlobalAwareAttentionMLP(nn.Module):
    def __init__(self, input_dim, n_parts=10, outsize=70):
        super().__init__()
        self.num_parts = n_parts
        hdim = 256

        self.global_mlp = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim)
        )

        self.attn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim + hdim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            )
            for _ in range(self.num_parts)
        ])

        self.parts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.2),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, outsize)
            )
            for _ in range(self.num_parts)
        ])
    
    def forward(self, x):
        global_context = self.global_mlp(x)

        outputs = []
        for attn_layer, mlp in zip(self.attn_layers, self.parts):
            combined = torch.cat([x, global_context], dim=-1)
            weights = attn_layer(combined)
            weighted_x = x * weights
            part_out = mlp(weighted_x)
            outputs.append(part_out)
        
        coeffs = torch.cat(outputs, dim=-1)
        return coeffs


def get_sinusoidal_positional_embeddings(n_positions, dim):
    pe = torch.zeros(n_positions, dim)
    position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape: (1, n_positions, dim)


class SelfAttentionAttentionMLP(nn.Module):
    def __init__(self, input_dim, n_parts=10, outsize=70):
        super().__init__()
        self.num_parts = n_parts
        hdim = 256

        self.parts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, hdim),
                nn.LeakyReLU(0.1),
                nn.Linear(hdim, outsize)
            )
            for _ in range(self.num_parts)
        ])

        self.self_attn = nn.MultiheadAttention(embed_dim=outsize, num_heads=7, batch_first=True)
        self.fc_out = nn.Linear(self.num_parts * outsize, self.num_parts * outsize)
        self.register_buffer('pos_embedding', get_sinusoidal_positional_embeddings(n_parts, outsize))

    def forward(self, x):
        part_outputs = []
        # Compute output for each independent part.
        for part in self.parts:
            part_out = part(x)  # shape: (batch_size, outsize)
            part_outputs.append(part_out)
        
        # Stack part outputs along a new dimension: (batch_size, num_parts, outsize)
        parts_out = torch.stack(part_outputs, dim=1)
        
        parts_out = parts_out + self.pos_embedding
    
        attn_output, attn_output_weights = self.self_attn(parts_out, parts_out, parts_out)
        # (attn_output has shape: (batch_size, num_parts, outsize))
        
        # Optional: Add a residual connection if desired:
        # attn_output = attn_output + parts_out
        
        attn_output_flat = attn_output.reshape(attn_output.size(0), -1)
        output = self.fc_out(attn_output_flat)
        
        return output
