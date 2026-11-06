
import os
import torch
from torch.utils.data import DataLoader

from config import (
    TRAIN_DIR,
    VALID_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    NUM_WORKERS,
    PIN_MEMORY,
    USE_AMP,
    DEVICE,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    PRINT_INTERVAL,
    SAVE_INTERVAL,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    DIM_FEEDFORWARD,
    MAX_SEQ_LEN,
    DROPOUT
)

from dataset import MidiDataset
from model import MidiTransformer
from losses import MidiLoss


def save_checkpoint(path, model, optimizer, epoch, val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }

    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    use_amp
):
    model.train()

    total_loss = 0.0
    pitch_loss_sum = 0.0
    velocity_loss_sum = 0.0
    duration_loss_sum = 0.0

    for x, targets, valid_mask in dataloader:

        x = x.to(DEVICE, non_blocking=True)

        targets = {
            key: value.to(DEVICE, non_blocking=True)
            for key, value in targets.items()
        }

        valid_mask = valid_mask.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp
        ):
            predictions = model(x)

            loss, losses = criterion(
                predictions,
                targets,
                valid_mask
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pitch_loss_sum += losses["pitch"].item()
        velocity_loss_sum += losses["velocity"].item()
        duration_loss_sum += losses["duration"].item()

    num_batches = len(dataloader)

    return {
        "total": total_loss / num_batches,
        "pitch": pitch_loss_sum / num_batches,
        "velocity": velocity_loss_sum / num_batches,
        "duration": duration_loss_sum / num_batches
    }


@torch.no_grad()
def validate(model, dataloader, criterion, use_amp):
    model.eval()

    total_loss = 0.0
    pitch_loss_sum = 0.0
    velocity_loss_sum = 0.0
    duration_loss_sum = 0.0

    for x, targets, valid_mask in dataloader:

        x = x.to(DEVICE, non_blocking=True)

        targets = {
            key: value.to(DEVICE, non_blocking=True)
            for key, value in targets.items()
        }

        valid_mask = valid_mask.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp
        ):
            predictions = model(x)

            loss, losses = criterion(
                predictions,
                targets,
                valid_mask
            )

        total_loss += loss.item()
        pitch_loss_sum += losses["pitch"].item()
        velocity_loss_sum += losses["velocity"].item()
        duration_loss_sum += losses["duration"].item()

    num_batches = len(dataloader)

    return {
        "total": total_loss / num_batches,
        "pitch": pitch_loss_sum / num_batches,
        "velocity": velocity_loss_sum / num_batches,
        "duration": duration_loss_sum / num_batches
    }


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    use_amp = USE_AMP and DEVICE.type == "cuda"

    print("=" * 60)
    print("MIDI TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Mixed precision: {use_amp}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Print interval: {PRINT_INTERVAL}")
    print(f"Save interval: {SAVE_INTERVAL}")
    print("=" * 60)

    # -------------------------------------------------
    # Dataset e DataLoaders
    # -------------------------------------------------

    train_dataset = MidiDataset(TRAIN_DIR)
    valid_dataset = MidiDataset(VALID_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0
    )

    # -------------------------------------------------
    # Model, Loss e Optimizer
    # -------------------------------------------------

    model = MidiTransformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = MidiLoss().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_val_loss = float("inf")

    num_parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Model parameters: {num_parameters:,}")
    print()

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------

    for epoch in range(1, NUM_EPOCHS + 1):

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            use_amp
        )

        valid_metrics = validate(
            model,
            valid_loader,
            criterion,
            use_amp
        )

        # Melhor checkpoint baseado na validação.
        is_best = valid_metrics["total"] < best_val_loss

        if is_best:
            best_val_loss = valid_metrics["total"]

            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                epoch,
                best_val_loss
            )

        # Salvamento periódico.
        if SAVE_INTERVAL > 0 and epoch % SAVE_INTERVAL == 0:
            save_checkpoint(
                LAST_MODEL_PATH,
                model,
                optimizer,
                epoch,
                valid_metrics["total"]
            )

        # Impressão periódica.
        should_print = (
            PRINT_INTERVAL > 0
            and epoch % PRINT_INTERVAL == 0
        )

        if should_print or epoch == NUM_EPOCHS:

            status = " [BEST]" if is_best else ""

            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}]{status}"
            )

            print(
                f"  Train: {train_metrics['total']:.6f} | "
                f"Pitch: {train_metrics['pitch']:.6f} | "
                f"Velocity: {train_metrics['velocity']:.6f} | "
                f"Duration: {train_metrics['duration']:.6f}"
            )

            print(
                f"  Valid: {valid_metrics['total']:.6f} | "
                f"Pitch: {valid_metrics['pitch']:.6f} | "
                f"Velocity: {valid_metrics['velocity']:.6f} | "
                f"Duration: {valid_metrics['duration']:.6f}"
            )

            print(
                f"  Best validation loss: {best_val_loss:.6f}"
            )

            print("-" * 60)

    # Salva o estado final, mesmo que não coincida
    # com o intervalo periódico.
    save_checkpoint(
        LAST_MODEL_PATH,
        model,
        optimizer,
        NUM_EPOCHS,
        valid_metrics["total"]
    )

    print("\nTreinamento concluído!")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Last model: {LAST_MODEL_PATH}")


if __name__ == "__main__":
    main()
