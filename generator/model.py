import torch
import torch.nn as nn


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=2048,
        d_model=256,
        n_heads=8,
        n_layers=6,
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

    def forward(self, tokens, padding_mask=None):
        # tokens: [batch, sequence_length]
        _, seq_len = tokens.shape
        if seq_len > self.block_size:
            raise ValueError("Sequência maior que block_size.")

        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]

        # True means that future positions are blocked.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool),
            diagonal=1
        )

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        x = self.final_norm(x)
        return self.output(x)


class MusicLSTM(nn.Module):
    """Baseline recorrente: mesmos tokens, mesma interface do MusicTransformer.

    forward(tokens) devolve logits [batch, seq, vocab], como o Transformer, e
    por isso funciona sem mudanças no laço de treino e no ConstrainedDecoder.
    Na geração, step() avança token a token carregando o estado (h, c), em vez
    de reprocessar a janela inteira a cada passo.
    """

    def __init__(
        self,
        vocab_size,
        block_size=1024,
        d_model=256,
        hidden_size=512,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()
        # block_size só limita as janelas de treino; a LSTM não tem
        # embedding posicional e aceita sequências de qualquer tamanho.
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, padding_mask=None, state=None, return_state=False):
        # O padding fica no fim de cada sequência (pad_sequence) e a LSTM é
        # causal, então ele não afeta as posições válidas; a loss já o ignora.
        x = self.embedding_dropout(self.token_embedding(tokens))
        x, state = self.lstm(x, state)
        logits = self.output(self.output_dropout(x))
        if return_state:
            return logits, state
        return logits

    @torch.no_grad()
    def step(self, tokens, state=None):
        """Processa só os tokens novos e devolve (logits do último, estado)."""
        logits, state = self.forward(tokens, state=state, return_state=True)
        return logits[0, -1, :], state


ARCHITECTURES = {
    "transformer": MusicTransformer,
    "lstm": MusicLSTM,
}


def build_model(arch, vocab_size, config):
    """Instancia a arquitetura pedida com os hiperparâmetros do checkpoint."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Arquitetura desconhecida: {arch}")
    return ARCHITECTURES[arch](vocab_size=vocab_size, **config)
