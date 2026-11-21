import argparse
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from vocabulary import Vocabulary
from dataset import MIDIDataset, collate_batch
from model import MusicTransformer


DATA_DIR = Path("data")
TRAIN_DATA_DIR = DATA_DIR / "train"
VALIDATION_DATA_DIR = DATA_DIR / "valid"

LEARNING_RATE = 3e-4
EFFECTIVE_BATCH_SIZE = 128
MICRO_BATCH_SIZE = 16
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
SEED = 42


def get_subset_paths(paths, fraction: float, seed: int = SEED):
    files = list(paths)
    random.Random(seed).shuffle(files)
    k = max(1, int(len(files) * fraction))
    return sorted(files[:k])


def main():
    parser = argparse.ArgumentParser(description="Controlled training script for Ablation Study (RTX 4060).")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name (e.g., ctx_256 or layers_2)")
    parser.add_argument("--block_size", type=int, default=1024, help="Context window size (e.g., 256 vs 1024)")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of Transformer layers (e.g., 2 vs 6)")
    parser.add_argument("--d_model", type=int, default=256, help="Hidden dimension size (e.g., 128 vs 256)")
    parser.add_argument("--train_fraction", type=float, default=0.20, help="Fixed training split fraction (default: 20%)")
    parser.add_argument("--epochs", type=int, default=30, help="Fixed number of epochs for fair comparison")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    output_dir = Path("output_ablation") / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.exp_name}_best.pt"

    all_train_paths = sorted(TRAIN_DATA_DIR.rglob("*.mid")) + sorted(TRAIN_DATA_DIR.rglob("*.midi"))
    train_paths = get_subset_paths(all_train_paths, args.train_fraction, seed=SEED)
    validation_paths = sorted(VALIDATION_DATA_DIR.rglob("*.mid")) + sorted(VALIDATION_DATA_DIR.rglob("*.midi"))

    if not train_paths or not validation_paths:
        raise FileNotFoundError("MIDI files not found in ./data/train or ./data/valid.")

    print(f"=== Ablation Experiment: {args.exp_name} ===")
    print(f"Selected training tracks: {len(train_paths)}/{len(all_train_paths)} ({args.train_fraction*100:.0f}%)")
    print(f"Validation tracks: {len(validation_paths)}")

    vocab = Vocabulary()
    train_dataset = MIDIDataset(train_paths, vocab, block_size=args.block_size, stride=args.block_size // 2)
    validation_dataset = MIDIDataset(validation_paths, vocab, block_size=args.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, vocab.pad_id),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, vocab.pad_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model_config = {
        "block_size": args.block_size,
        "d_model": args.d_model,
        "n_heads": 8,
        "n_layers": args.n_layers,
        "dropout": 0.1,
    }
    model = MusicTransformer(vocab_size=vocab.size, **model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler("cuda", enabled=not use_bf16)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
                logits = model(x, padding_mask=x.eq(vocab.pad_id))
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab.size),
                    y.reshape(-1),
                    ignore_index=vocab.pad_id,
                )
                scaled_loss = loss / ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()
            total_loss += loss.item()

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in validation_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    logits = model(x, padding_mask=x.eq(vocab.pad_id))
                    val_loss += F.cross_entropy(
                        logits.reshape(-1, vocab.size),
                        y.reshape(-1),
                        ignore_index=vocab.pad_id,
                    ).item()
        avg_val_loss = val_loss / len(validation_loader)
        history.append((epoch, avg_train_loss, avg_val_loss))

        print(f"[{args.exp_name}] Epoch {epoch:02d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab.size,
                **model_config,
                "validation_loss": avg_val_loss,
                "epoch": epoch,
            }, checkpoint_path)

    csv_path = output_dir / f"{args.exp_name}_loss.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for ep, tr, va in history:
            f.write(f"{ep},{tr:.6f},{va:.6f}\n")

    print(f"Done! Best Val Loss: {best_val_loss:.4f} | Saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
