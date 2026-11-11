
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import (
    TEST_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    USE_AMP,
    DEVICE,
    BEST_MODEL_PATH,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    DIM_FEEDFORWARD,
    MAX_SEQ_LEN,
    DROPOUT,
    NUM_OUTPUTS
)

from dataset import MidiDataset
from model import MidiTransformer
from losses import MidiLoss


def load_model(checkpoint_path):
    model = MidiTransformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "Unknown")
    else:
        model.load_state_dict(checkpoint)
        epoch = "Unknown"

    model.eval()

    return model, epoch


@torch.inference_mode()
def evaluate(model, dataloader, criterion, use_amp):
    total_loss = 0.0
    pitch_loss_sum = 0.0
    velocity_loss_sum = 0.0
    duration_loss_sum = 0.0

    total_samples = 0

    # Contadores de classificação de pitch.
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    # Erros de velocity e duration.
    velocity_error_sum = 0.0
    duration_error_sum = 0.0
    total_notes = 0

    log_duration_max = np.log1p(65535.0)

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

        batch_size = x.size(0)

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

        total_loss += loss.item() * batch_size
        pitch_loss_sum += losses["pitch"].item() * batch_size
        velocity_loss_sum += losses["velocity"].item() * batch_size
        duration_loss_sum += losses["duration"].item() * batch_size

        total_samples += batch_size

        # ---------------------------------------------
        # Pitch: precisão, recall e F1
        # ---------------------------------------------

        pitch_prob = torch.sigmoid(
            predictions["pitch_logits"]
        )

        pitch_pred = pitch_prob >= 0.5
        pitch_true = targets["pitch"] >= 0.5

        temporal_mask = valid_mask.unsqueeze(-1).expand_as(
            pitch_true
        )

        true_positives += (
            pitch_pred & pitch_true & temporal_mask
        ).sum().item()

        false_positives += (
            pitch_pred & ~pitch_true & temporal_mask
        ).sum().item()

        false_negatives += (
            ~pitch_pred & pitch_true & temporal_mask
        ).sum().item()

        true_negatives += (
            ~pitch_pred & ~pitch_true & temporal_mask
        ).sum().item()

        # ---------------------------------------------
        # Velocity e Duration: somente notas reais
        # ---------------------------------------------

        note_mask = pitch_true & temporal_mask

        num_notes = note_mask.sum().item()

        if num_notes > 0:
            velocity_pred = predictions["velocity"].clamp(0, 1)
            duration_pred = predictions["duration"].clamp(0, 1)

            velocity_true = targets["velocity"]
            duration_true = targets["duration"]

            # Converte velocity normalizada para escala MIDI.
            velocity_pred = velocity_pred * 127.0
            velocity_true = velocity_true * 127.0

            # Converte duration normalizada de volta
            # para a quantidade de intervalos temporais.
            duration_pred = (
                torch.expm1(duration_pred * log_duration_max)
            )

            duration_true = (
                torch.expm1(duration_true * log_duration_max)
            )

            velocity_errors = torch.abs(
                velocity_pred - velocity_true
            )

            duration_errors = torch.abs(
                duration_pred - duration_true
            )

            velocity_error_sum += (
                velocity_errors * note_mask
            ).sum().item()

            duration_error_sum += (
                duration_errors * note_mask
            ).sum().item()

            total_notes += num_notes

    # ---------------------------------------------
    # Métricas finais
    # ---------------------------------------------

    precision = true_positives / max(
        true_positives + false_positives, 1
    )

    recall = true_positives / max(
        true_positives + false_negatives, 1
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    accuracy = (
        (true_positives + true_negatives)
        / max(
            true_positives + true_negatives
            + false_positives + false_negatives,
            1
        )
    )

    velocity_mae = (
        velocity_error_sum / total_notes
        if total_notes > 0 else 0.0
    )

    duration_mae = (
        duration_error_sum / total_notes
        if total_notes > 0 else 0.0
    )

    return {
        "loss": total_loss / max(total_samples, 1),
        "pitch_loss": pitch_loss_sum / max(total_samples, 1),
        "velocity_loss": velocity_loss_sum / max(total_samples, 1),
        "duration_loss": duration_loss_sum / max(total_samples, 1),
        "pitch_precision": precision,
        "pitch_recall": recall,
        "pitch_f1": f1,
        "pitch_accuracy": accuracy,
        "velocity_mae": velocity_mae,
        "duration_mae": duration_mae,
        "total_notes": total_notes
    }


def main():
    use_amp = USE_AMP and DEVICE.type == "cuda"

    print("=" * 60)
    print("MIDI TRANSFORMER EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Mixed precision: {use_amp}")
    print(f"Checkpoint: {BEST_MODEL_PATH}")
    print("=" * 60)

    # Dataset de teste.
    test_dataset = MidiDataset(TEST_DIR)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0
    )

    # Carrega o modelo.
    model, epoch = load_model(BEST_MODEL_PATH)

    criterion = MidiLoss().to(DEVICE)

    num_parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nCheckpoint epoch: {epoch}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Model parameters: {num_parameters:,}")
    print("\nEvaluating...\n")

    metrics = evaluate(
        model,
        test_loader,
        criterion,
        use_amp
    )

    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print("\nLoss:")
    print(f"  Total:    {metrics['loss']:.6f}")
    print(f"  Pitch:    {metrics['pitch_loss']:.6f}")
    print(f"  Velocity: {metrics['velocity_loss']:.6f}")
    print(f"  Duration: {metrics['duration_loss']:.6f}")

    print("\nPitch:")
    print(f"  Precision: {metrics['pitch_precision']:.6f}")
    print(f"  Recall:    {metrics['pitch_recall']:.6f}")
    print(f"  F1-score:  {metrics['pitch_f1']:.6f}")
    print(f"  Accuracy:  {metrics['pitch_accuracy']:.6f}")

    print("\nVelocity:")
    print(f"  MAE: {metrics['velocity_mae']:.4f} / 127")

    print("\nDuration:")
    print(
        f"  MAE: {metrics['duration_mae']:.4f} "
        "intervalos de 1/32 de semínima"
    )

    print(f"\nTotal de notas avaliadas: {metrics['total_notes']:,}")

    print("\nAvaliação concluída!")


if __name__ == "__main__":
    main()
