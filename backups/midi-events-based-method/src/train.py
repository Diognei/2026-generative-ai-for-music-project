from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vocabulary import Vocabulary
from dataset import MIDIDataset, collate_batch
from model import MusicTransformer


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"

BLOCK_SIZE = 512
BATCH_SIZE = 4096
EPOCHS = 20000
LEARNING_RATE = 3e-4


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    midi_paths = sorted(DATA_DIR.glob("*.mid"))
    midi_paths += sorted(DATA_DIR.glob("*.midi"))

    if not midi_paths:
        raise FileNotFoundError(
            "Nenhum MIDI encontrado em ./data/. "
            "Coloque o arquivo em ./data/song1.mid"
        )

    print(f"MIDI(s) encontrados: {len(midi_paths)}")
    vocab = Vocabulary()
    vocab.save(OUTPUT_DIR / "vocabulary.json")

    dataset = MIDIDataset(midi_paths, vocab, block_size=BLOCK_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, vocab.pad_id)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(
        vocab_size=vocab.size,
        block_size=BLOCK_SIZE
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab.size}")
    print(f"Training windows: {len(dataset)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
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

        average_loss = total_loss / len(loader)

        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS} | Loss: {average_loss:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab.size,
        "block_size": BLOCK_SIZE,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "dropout": 0.1
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Modelo salvo em: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
