import argparse
import random
from pathlib import Path
import torch

from vocabulary import Vocabulary
from model import MusicTransformer
from midi_events import tokens_to_midi
from align_midi import fix_midi
from generate import (
    generate,
    TEMPERATURE,
    MAX_NEW_TOKENS,
    ALIGN_FINAL_MIDI,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generates N MIDI samples in batch from an ablation checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the *_best.pt checkpoint file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory where .mid files will be saved")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of pieces to generate (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed for reproducibility")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vocab = Vocabulary()

    if checkpoint["vocab_size"] != vocab.size:
        raise RuntimeError("Vocabulary mismatch between checkpoint and current codebase.")

    model = MusicTransformer(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"=== Batch Generation: {args.checkpoint.name} ===")
    print(f"Device: {device} | block_size: {checkpoint['block_size']} | Samples: {args.num_samples}")

    generated_count = 0
    attempt = 0

    while generated_count < args.num_samples:
        attempt += 1
        current_seed = args.seed + attempt
        random.seed(current_seed)
        torch.manual_seed(current_seed)

        try:
            tokens = generate(
                model=model,
                vocab=vocab,
                device=device,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            raw_midi_path = args.output_dir / f"sample_{generated_count + 1:03d}_raw.mid"
            tokens_to_midi(tokens, vocab, raw_midi_path)

            if ALIGN_FINAL_MIDI:
                aligned_path = Path(fix_midi(raw_midi_path))
                final_path = args.output_dir / f"sample_{generated_count + 1:03d}.mid"
                aligned_path.replace(final_path)
                raw_midi_path.unlink(missing_ok=True)
            else:
                final_path = args.output_dir / f"sample_{generated_count + 1:03d}.mid"
                raw_midi_path.replace(final_path)

            generated_count += 1
            print(f"[{generated_count:03d}/{args.num_samples:03d}] Generated: {final_path.name} ({len(tokens)} tokens)", flush=True)

        except Exception as exc:
            print(f"[Warning] Attempt {attempt} failed ({exc}), retrying...", flush=True)

    print(f"\nDone! {generated_count} MIDI files ready in: {args.output_dir}")


if __name__ == "__main__":
    main()
