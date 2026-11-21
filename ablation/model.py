# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard Python math library: used here for square root scaling of embeddings.
import math

# Core PyTorch library: provides Tensor data structures and GPU acceleration.
import torch

# PyTorch Neural Network module: contains building blocks like Linear, Embedding,
# LayerNorm, and TransformerEncoderLayer (permitted by the assignment rules).
import torch.nn as nn

# PyTorch Functional API: provides stateless functions like softmax and relu.
import torch.nn.functional as F


# ==============================================================================
# MODEL CLASS DEFINITION
# ==============================================================================
class MusicTransformer(nn.Module):
    """
    A Decoder-Only Autoregressive Transformer (GPT-style) built using native PyTorch
    primitives. It predicts the next musical token given a sequence of past tokens.
    
    Inherits from `nn.Module`, which is the base class for all neural network modules
    in PyTorch (handles parameter tracking, GPU transfer via `.to('cuda')`, etc.).
    """

    # --------------------------------------------------------------------------
    # 1. CONSTRUCTOR (__init__): Declares all learnable layers and fixed buffers
    # --------------------------------------------------------------------------
    def __init__(
        self,
        vocab_size: int,          # Total number of unique MIDI tokens in the vocabulary.
        d_model: int = 256,       # Dimensionality of the internal hidden vectors (embeddings).
        nhead: int = 8,           # Number of parallel self-attention heads (must divide d_model).
        num_layers: int = 4,      # Number of stacked Transformer blocks (depth of the network).
        dim_feedforward: int = 1024, # Hidden dimension inside the position-wise MLP (usually 4 * d_model).
        max_seq_len: int = 512,   # Maximum context window length (number of past tokens visible).
        dropout: float = 0.1,     # Probability of zeroing activations during training to prevent overfitting.
    ):
        # Calls the parent `nn.Module` initializer so PyTorch registers internal layers properly.
        super().__init__()

        # Store hyperparameters as instance attributes for later use in `forward` and `generate`.
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # --- LAYER GROUP 1: INPUT REPRESENTATION (TOKEN + POSITIONAL EMBEDDINGS) ---
        # `nn.Embedding(num_embeddings, embedding_dim)` acts as a learnable lookup table.
        # Input: integer token IDs [Batch, Seq_Len] -> Output: dense vectors [Batch, Seq_Len, d_model].
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model
        )

        # Since Transformers process all tokens in parallel without recurrence, they have no
        # built-in sense of order. `pos_embedding` learns a unique vector for each time step
        # from index `0` up to `max_seq_len - 1`.
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_seq_len, 
            embedding_dim=d_model
        )

        # `nn.Dropout` randomly zeroes elements with probability `dropout` during `.train()`
        # mode (and automatically disables itself during `.eval()` mode).
        self.dropout = nn.Dropout(p=dropout)

        # --- LAYER GROUP 2: CAUSAL ATTENTION MASK (AUTOREGRESSIVE CONSTRAINT) ---
        # Creates a 2D square matrix of shape [max_seq_len, max_seq_len] filled with ones.
        ones_matrix = torch.ones(max_seq_len, max_seq_len)

        # `torch.triu(..., diagonal=1)` keeps only the upper triangle strictly ABOVE the main diagonal,
        # setting the diagonal and lower triangle to 0. Multiplying by `-inf` turns future positions
        # into negative infinity. When added to attention scores before Softmax, `-inf` becomes `0.0`
        # probability, strictly preventing token `t` from seeing future tokens `t+1, t+2, ...`.
        causal_mask = torch.triu(ones_matrix * float("-inf"), diagonal=1)

        # `register_buffer` saves `causal_mask` inside the model state (so `.to('cuda')` moves it
        # to the GPU automatically), while telling PyTorch it is a fixed constant, NOT a learnable weight.
        self.register_buffer("causal_mask", causal_mask)

        # --- LAYER GROUP 3: TRANSFORMER DECODER STACK ---
        # `nn.TransformerEncoderLayer` implements one complete Transformer block:
        #   1. Multi-Head Self-Attention (learns relationships across musical time steps)
        #   2. Residual connections + Layer Normalization (`nn.LayerNorm`)
        #   3. Two-layer Feed-Forward Network (Linear -> GELU -> Dropout -> Linear)
        # Note: Even though PyTorch names this class "EncoderLayer", passing a `causal_mask`
        # in `forward()` turns it into a causal Decoder block (identical to GPT architecture).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                 # Input and output feature dimension.
            nhead=nhead,                     # Number of attention heads splitting `d_model`.
            dim_feedforward=dim_feedforward, # Inner dimension of the feed-forward sub-network.
            dropout=dropout,                 # Dropout rate inside attention and feed-forward layers.
            activation="gelu",               # Gaussian Error Linear Unit (smoother than ReLU, standard in GPT).
            batch_first=True,                # Expects tensors shaped as [Batch, Seq_Len, d_model].
            norm_first=True,                 # Pre-LayerNorm: normalizes inputs BEFORE attention/MLP (stabilizes training).
        )

        # `nn.TransformerEncoder` clones and stacks `encoder_layer` sequentially `num_layers` times.
        self.transformer_blocks = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Final `nn.LayerNorm` applied after the last Transformer block (required when `norm_first=True`).
        self.final_norm = nn.LayerNorm(normalized_shape=d_model)

        # --- LAYER GROUP 4: OUTPUT PROJECTION HEAD ---
        # `nn.Linear(in_features, out_features)` applies an affine transformation `y = xW^T`.
        # Projects each hidden state vector of size `d_model` back into raw unnormalized scores
        # (`logits`) over the entire `vocab_size` to predict the next token.
        self.output_head = nn.Linear(
            in_features=d_model, 
            out_features=vocab_size, 
            bias=False
        )

        # Weight Tying: Shares the exact same weight matrix between `token_embedding` and `output_head`.
        # Semantically, tokens with similar embeddings should also receive similar output logits.
        # Practically, this reduces total parameter count and saves GPU VRAM on your RTX 4060.
        self.output_head.weight = self.token_embedding.weight

        # Call custom weight initialization method to ensure clean starting gradients.
        self._init_weights()

    # --------------------------------------------------------------------------
    # 2. WEIGHT INITIALIZATION HELPER
    # --------------------------------------------------------------------------
    def _init_weights(self):
        """
        Initializes Linear and Embedding weights using a small Normal distribution (std=0.02),
        following standard GPT initialization practices to prevent exploding activations at step 0.
        """
        # `self.modules()` recursively iterates over every sub-layer defined inside `__init__`.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Fill weight tensor in-place with values drawn from N(0, 0.02^2).
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    # Zero out bias vectors if present.
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Fill embedding lookup tables with values drawn from N(0, 0.02^2).
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # --------------------------------------------------------------------------
    # 3. FORWARD PASS: Defines how input data flows through the network
    # --------------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass for both training (parallel over all time steps) and
        inference (evaluating a context window).

        Args:
            input_ids (torch.Tensor): Integer tensor of token IDs, shape [batch_size, seq_len].

        Returns:
            logits (torch.Tensor): Raw prediction scores for every next token,
                                   shape [batch_size, seq_len, vocab_size].
        """
        # Unpack the batch size (B) and current sequence length (T) from the input tensor shape.
        batch_size, seq_len = input_ids.shape

        # Safety check: ensures the input sequence fits inside our positional embedding table.
        assert seq_len <= self.max_seq_len, (
            f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})."
        )

        # Step 1: Build position indices `[0, 1, 2, ..., seq_len - 1]` on the same device (CPU/GPU)
        # as `input_ids`, then add a batch dimension via `.unsqueeze(0)` -> shape [1, seq_len].
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        # Step 2: Look up token embeddings and scale by `sqrt(d_model)` (from "Attention Is All You Need")
        # so token signal magnitude balances properly against positional embeddings.
        # Shape: [batch_size, seq_len, d_model].
        tok_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        # Look up positional embeddings -> shape [1, seq_len, d_model] (broadcasts across batch_size).
        pos_emb = self.pos_embedding(positions)

        # Sum token + positional embeddings element-wise and apply dropout.
        x = self.dropout(tok_emb + pos_emb)

        # Step 3: Slice the pre-computed causal mask to match the exact current `seq_len`.
        # Shape: [seq_len, seq_len].
        mask = self.causal_mask[:seq_len, :seq_len]

        # Step 4: Pass `x` through all stacked Transformer blocks.
        # `is_causal=True` enables PyTorch's fast FlashAttention/SDPA causal kernel when available.
        # Shape remains: [batch_size, seq_len, d_model].
        x = self.transformer_blocks(x, mask=mask, is_causal=True)

        # Apply final layer normalization.
        x = self.final_norm(x)

        # Step 5: Project hidden vectors to vocabulary logits.
        # Output shape: [batch_size, seq_len, vocab_size].
        logits = self.output_head(x)

        return logits

    # --------------------------------------------------------------------------
    # 4. AUTOREGRESSIVE GENERATION LOOP (INFERENCE / SAMPLING)
    # --------------------------------------------------------------------------
    # `@torch.no_grad()` disables gradient calculation and autograd graph construction,
    # drastically reducing VRAM usage and speeding up token-by-token generation.
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,       # Starting seed tokens, shape [batch_size, prompt_len].
        max_new_tokens: int = 512,      # Number of new musical tokens to generate autoregressively.
        temperature: float = 1.0,       # Controls randomness: <1.0 = conservative, >1.0 = creative/chaotic.
        top_k: int = 0,                 # If > 0, keeps only the top K most likely tokens before sampling.
        top_p: float = 0.9,             # Nucleus sampling: keeps smallest set of tokens with cumulative prob >= top_p.
        eos_token_id: int | None = None # Optional End-Of-Sequence token ID to stop generation early.
    ) -> torch.Tensor:
        """
        Generates new token sequences one token at a time, conditioning each step on all
        previously generated tokens.
        """
        # Set module to evaluation mode (disables Dropout layers for deterministic forward passes).
        self.eval()

        # Initialize the running sequence with the provided prompt tokens.
        generated = prompt_ids

        # Loop `max_new_tokens` times, appending one newly sampled token per iteration.
        for _ in range(max_new_tokens):
            # Sliding Window Crop: If `generated` becomes longer than `max_seq_len`, keep only
            # the most recent `max_seq_len` tokens so `pos_embedding` never goes out of bounds.
            context = generated[:, -self.max_seq_len :]

            # Run forward pass on `context` -> shape [batch_size, context_len, vocab_size].
            all_logits = self.forward(context)

            # Extract ONLY the logits corresponding to the LAST time step (`-1`), which represents
            # the model's prediction for the next unseen token -> shape [batch_size, vocab_size].
            next_token_logits = all_logits[:, -1, :]

            # --- SAMPLING STRATEGY A: TEMPERATURE SCALING ---
            # Divide logits by temperature (clamped to `1e-5` to prevent division by zero).
            # Lower temperature sharpens peaks (more deterministic); higher flattens distribution.
            next_token_logits = next_token_logits / max(temperature, 1e-5)

            # --- SAMPLING STRATEGY B: TOP-K FILTERING ---
            if top_k > 0:
                # Find the K largest logit values for each sequence in the batch.
                k = min(top_k, next_token_logits.size(-1))
                top_k_values, _ = torch.topk(next_token_logits, k=k)
                # Threshold is the K-th highest logit (`top_k_values[:, [-1]]`).
                # Set all logits strictly below this threshold to `-inf` (0 probability after softmax).
                next_token_logits[next_token_logits < top_k_values[:, [-1]]] = float("-inf")

            # --- SAMPLING STRATEGY C: TOP-P (NUCLEUS) FILTERING ---
            if 0.0 < top_p < 1.0:
                # Sort logits in descending order from most probable to least probable.
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)

                # Convert sorted logits into probabilities via Softmax and compute cumulative sum.
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Create a boolean mask identifying tokens whose cumulative probability exceeds `top_p`.
                sorted_indices_to_remove = cumulative_probs > top_p

                # Shift the mask right by 1 position (`[..., 1:] = [..., :-1]`) and force index 0 to False.
                # This guarantees that the single most likely token is ALWAYS kept, even if its
                # individual probability alone exceeds `top_p`.
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter the boolean mask back from sorted order to the original vocabulary index order.
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )

                # Mask out rejected tokens with `-inf`.
                next_token_logits[indices_to_remove] = float("-inf")

            # --- FINAL STEP: CONVERT LOGITS TO PROBABILITIES AND SAMPLE ---
            # `F.softmax` normalizes logits across the vocabulary dimension (`dim=-1`) so they sum to 1.0.
            probs = F.softmax(next_token_logits, dim=-1)

            # `torch.multinomial` randomly draws 1 token index per batch item weighted by `probs`.
            next_token = torch.multinomial(probs, num_samples=1)  # Shape: [batch_size, 1]

            # Concatenate the newly sampled token onto the right end of `generated` along `dim=1`.
            generated = torch.cat([generated, next_token], dim=1)

            # Early stopping: if an `<EOS>` token ID was provided and every sequence in the batch
            # just emitted `<EOS>`, terminate the loop early.
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        # Return the complete sequence containing the original prompt plus generated tokens.
        return generated


# ==============================================================================
# STANDALONE SMOKE TEST (Executed only when running `python model.py` directly)
# ==============================================================================
if __name__ == "__main__":
    # Automatically select NVIDIA GPU ('cuda') if available, otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running smoke test on device: {device}")

    # Instantiate a small test configuration of the MusicTransformer and move weights to `device`.
    model = MusicTransformer(
        vocab_size=512,
        d_model=256,
        nhead=8,
        num_layers=4,
        max_seq_len=512,
    ).to(device)

    # Create a dummy batch of 2 sequences, each containing 64 random token IDs in [0, 511].
    dummy_input = torch.randint(low=0, high=512, size=(2, 64), device=device)

    # Test 1: Training forward pass (verifies tensor shapes and causal masking).
    logits = model(dummy_input)
    print("Training logits shape:", logits.shape)  # Expected: torch.Size([2, 64, 512])

    # Test 2: Autoregressive generation (feeds 4 seed tokens and generates 20 new tokens).
    generated_seq = model.generate(
        prompt_ids=dummy_input[:, :4],
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
    )
    print("Generated sequence shape:", generated_seq.shape)  # Expected: torch.Size([2, 24])
