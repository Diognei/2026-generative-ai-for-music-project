import torch
import torch.nn as nn


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=512,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        # tokens: [batch, sequence_length]
        batch_size, seq_len = tokens.shape
        if seq_len > self.block_size:
            raise ValueError("Sequência maior que block_size.")

        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]

        # True significa que a posição futura fica bloqueada.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool),
            diagonal=1
        )

        x = self.transformer(x, mask=causal_mask)
        x = self.final_norm(x)
        return self.output(x)
