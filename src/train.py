from pathlib import Path
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vocabulary import Vocabulary
from dataset import MIDIDataset, collate_batch
from model import MusicTransformer


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"

BLOCK_SIZE = 1024
BATCH_SIZE = 1024
EPOCHS = 100000
LEARNING_RATE = 3e-4
SAVE_INTERVAL = 1000
PRINT_INTERVAL = 100
MODEL_CONFIG = {
    "block_size": BLOCK_SIZE,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "dropout": 0.1,
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    midi_paths = sorted(DATA_DIR.rglob("*.mid"))
    midi_paths += sorted(DATA_DIR.rglob("*.midi"))

    if not midi_paths:
        raise FileNotFoundError(
            "Nenhum MIDI encontrado em ./data/. "
            "Coloque o arquivo em ./data/song1.mid"
        )

    print(f"MIDI(s) encontrados: {len(midi_paths)}")
    vocab = Vocabulary()
    vocab.save(OUTPUT_DIR / "vocabulary.json")

    random.Random(42).shuffle(midi_paths)
    split = max(1, round(len(midi_paths) * 0.9))
    train_paths = midi_paths[:split]
    validation_paths = midi_paths[split:] or train_paths[-1:]
    train_dataset = MIDIDataset(train_paths, vocab, block_size=BLOCK_SIZE, stride=BLOCK_SIZE // 2)
    validation_dataset = MIDIDataset(validation_paths, vocab, block_size=BLOCK_SIZE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, vocab.pad_id)
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, vocab.pad_id)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(vocab_size=vocab.size, **MODEL_CONFIG).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab.size}")
    print(f"Training windows: {len(train_dataset)}")
    print(f"Validation windows: {len(validation_dataset)}")
    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x, padding_mask=x.eq(vocab.pad_id))
            loss = F.cross_entropy(
                logits.reshape(-1, vocab.size),
                y.reshape(-1),
                ignore_index=vocab.pad_id
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for x, y in validation_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x, padding_mask=x.eq(vocab.pad_id))
                validation_loss += F.cross_entropy(
                    logits.reshape(-1, vocab.size), y.reshape(-1),
                    ignore_index=vocab.pad_id
                ).item()
        validation_loss /= len(validation_loader)

        if epoch % PRINT_INTERVAL == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS} | Train: {average_loss:.6f} | Val: {validation_loss:.6f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab.size,
                **MODEL_CONFIG,
                "validation_loss": validation_loss,
            }, CHECKPOINT_PATH)

        if epoch % SAVE_INTERVAL == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab.size,
                **MODEL_CONFIG,
                "validation_loss": validation_loss,
                "epoch": epoch,
            }, OUTPUT_DIR / f"model_epoch_{epoch}.pt")
            print(f"Checkpoint periódico salvo: model_epoch_{epoch}.pt")

    print(
        f"Melhor validation loss: {best_validation_loss:.6f} | "
        f"Checkpoint salvo em: {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()
