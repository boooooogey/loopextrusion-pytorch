"""Implementation of left right transformer for left and right parameters in the dlem model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = RotaryPositionalEmbeddings(d_model//num_heads)

    def forward(self, x):
        # Self-attention with residual connection
        num_heads, d_model = self.num_heads, self.d_model
        x_rope = self.rope(x.view(x.size(0),
                                  x.size(1),
                                  num_heads, d_model//num_heads)).view(x.size(0), x.size(1), -1)
        attn_output, _ = self.self_attn(x_rope, x_rope, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward network with residual connection
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbeddings(d_model//num_heads)
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, x, memory, causal_mask=None):
        # Self-attention (masked)
        num_heads, d_model = self.num_heads, self.d_model
        x_rope = self.rope(x.view(x.size(0),
                                  x.size(1),
                                  num_heads, d_model//num_heads)).view(x.size(0), x.size(1), -1)
        attn_output, _ = self.self_attn(x_rope, x_rope, x, attn_mask=causal_mask, is_causal=True)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention (attends to encoder output)
        x_rope = self.rope(x.view(x.size(0),
                                  x.size(1),
                                  num_heads, d_model//num_heads)).view(x.size(0), x.size(1), -1)
        memory_rope = self.rope(memory.view(memory.size(0),
                                  memory.size(1),
                                  num_heads,
                                  d_model//num_heads)).view(memory.size(0), memory.size(1), -1)
        attn_output, _ = self.cross_attn(x_rope, memory_rope, memory,
                                         attn_mask=causal_mask, is_causal=True)
        x = self.norm2(x + self.dropout(attn_output))

        # Feedforward network
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=16, num_heads=8, num_layers=3, dim_feedforward=24, dropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     num_heads,
                                     dim_feedforward,
                                     dropout) for _ in range(num_layers)]
        )
        self.decoder_layers_left = nn.ModuleList(
            [TransformerDecoderLayer(d_model,
                                     num_heads,
                                     dim_feedforward,
                                     dropout) for _ in range(num_layers)]
        )

        self.decoder_layers_right = nn.ModuleList(
            [TransformerDecoderLayer(d_model,
                                     num_heads,
                                     dim_feedforward,
                                     dropout) for _ in range(num_layers)]
        )

    def generate_mask(self, seq_len, is_right=False):
        causal_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)  # Transpose to flip causality
        if is_right:
            causal_mask = causal_mask.T
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))  # Convert to -inf
        causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)  # Convert 1s to 0s
        return causal_mask

    def forward(self, memory, out_left, out_right):
        # Create causal masks
        assert memory.size(1) == out_left.size(1)
        assert memory.size(1) == out_right.size(1)
        left_mask = self.generate_mask(out_left.size(1))
        right_mask = self.generate_mask(out_right.size(1), is_right=True)
        left_mask = left_mask.to(memory.device)
        right_mask = right_mask.to(memory.device)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            memory = layer(memory)

        # Pass through decoder layers
        for layer_left, layer_right in zip(self.decoder_layers_left, self.decoder_layers_right):
            out_left = layer_left(out_left, memory, causal_mask=left_mask)
            out_right = layer_right(out_right, memory, causal_mask=right_mask)

        return out_left, out_right
