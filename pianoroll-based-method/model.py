
import torch
import torch.nn as nn


class MidiTransformer(nn.Module):
    def __init__(
        self,
        num_tracks=11,
        num_pitches=60,
        features_per_value=3,
        d_model=256,
        num_heads=8,
        num_layers=6,
        dim_feedforward=1024,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()

        self.num_tracks = num_tracks
        self.num_pitches = num_pitches
        self.num_outputs = num_tracks * num_pitches
        self.input_dim = self.num_outputs * features_per_value
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Projeção dos dados de entrada para o espaço latente.
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # Embeddings posicionais aprendidos.
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

        # Blocos Transformer com máscara causal.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Cabeças independentes para cada tipo de informação musical.
        self.pitch_head = nn.Linear(d_model, self.num_outputs)
        self.velocity_head = nn.Linear(d_model, self.num_outputs)
        self.duration_head = nn.Linear(d_model, self.num_outputs)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor com shape (batch_size, sequence_length, 1980).

        Returns:
            Dicionário contendo:
                pitch_logits: (B, T, 660)
                velocity:     (B, T, 660)
                duration:     (B, T, 660)
        """

        if x.ndim != 3:
            raise ValueError(
                f"Esperado tensor 3D (B, T, F), recebido: {tuple(x.shape)}"
            )

        batch_size, seq_len, input_dim = x.shape

        if input_dim != self.input_dim:
            raise ValueError(
                f"Esperadas {self.input_dim} features, recebidas {input_dim}."
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequência com {seq_len} posições excede "
                f"max_seq_len={self.max_seq_len}."
            )

        # Projeção de entrada e informação posicional.
        x = self.input_projection(x)
        x = x + self.position_embedding[:, :seq_len, :]
        x = self.dropout(x)

        # Máscara causal: impede acesso a posições futuras.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        x = self.transformer(x, mask=causal_mask)

        # Predições independentes.
        pitch_logits = self.pitch_head(x)
        velocity = self.velocity_head(x)
        duration = self.duration_head(x)

        return {
            "pitch_logits": pitch_logits,
            "velocity": velocity,
            "duration": duration
        }
