
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    PITCH_LOSS_WEIGHT,
    VELOCITY_LOSS_WEIGHT,
    DURATION_LOSS_WEIGHT
)


class MidiLoss(nn.Module):
    def __init__(
        self,
        pitch_weight=PITCH_LOSS_WEIGHT,
        velocity_weight=VELOCITY_LOSS_WEIGHT,
        duration_weight=DURATION_LOSS_WEIGHT,
        pitch_pos_weight=None
    ):
        super().__init__()

        self.pitch_weight = pitch_weight
        self.velocity_weight = velocity_weight
        self.duration_weight = duration_weight

        if pitch_pos_weight is not None:
            self.register_buffer(
                "pitch_pos_weight",
                torch.tensor(float(pitch_pos_weight))
            )
        else:
            self.pitch_pos_weight = None

    def forward(self, predictions, targets, valid_mask):
        """
        Args:
            predictions:
                Dicionário retornado pelo MidiTransformer:
                    pitch_logits: (B, T, 660)
                    velocity:     (B, T, 660)
                    duration:     (B, T, 660)

            targets:
                Dicionário retornado pelo MidiDataset:
                    pitch:    (B, T, 660)
                    velocity: (B, T, 660)
                    duration: (B, T, 660)

            valid_mask:
                (B, T), indicando posições temporais válidas.

        Returns:
            total_loss, losses
        """

        pitch_logits = predictions["pitch_logits"]
        velocity_pred = predictions["velocity"]
        duration_pred = predictions["duration"]

        pitch_target = targets["pitch"]
        velocity_target = targets["velocity"]
        duration_target = targets["duration"]

        # Máscara temporal: ignora posições de padding.
        temporal_mask = valid_mask.unsqueeze(-1)

        temporal_mask = temporal_mask.expand_as(pitch_target)

        # -------------------------------------------------
        # 1. Pitch loss: classificação binária multilabel
        # -------------------------------------------------

        pitch_loss_raw = F.binary_cross_entropy_with_logits(
            pitch_logits,
            pitch_target,
            reduction="none",
            pos_weight=self.pitch_pos_weight
        )

        pitch_loss = (
            pitch_loss_raw * temporal_mask
        ).sum() / temporal_mask.sum().clamp(min=1)

        # -------------------------------------------------
        # 2. Velocity loss: regressão Smooth L1
        # -------------------------------------------------

        # Só considera posições onde existe uma nota real.
        note_mask = (
            (pitch_target > 0.5) & temporal_mask
        )

        velocity_loss_raw = F.smooth_l1_loss(
            velocity_pred,
            velocity_target,
            reduction="none"
        )

        if note_mask.any():
            velocity_loss = (
                velocity_loss_raw * note_mask
            ).sum() / note_mask.sum()
        else:
            velocity_loss = velocity_pred.sum() * 0.0

        # -------------------------------------------------
        # 3. Duration loss: regressão Smooth L1
        # -------------------------------------------------

        duration_loss_raw = F.smooth_l1_loss(
            duration_pred,
            duration_target,
            reduction="none"
        )

        if note_mask.any():
            duration_loss = (
                duration_loss_raw * note_mask
            ).sum() / note_mask.sum()
        else:
            duration_loss = duration_pred.sum() * 0.0

        # -------------------------------------------------
        # 4. Loss total
        # -------------------------------------------------

        total_loss = (
            self.pitch_weight * pitch_loss
            + self.velocity_weight * velocity_loss
            + self.duration_weight * duration_loss
        )

        losses = {
            "pitch": pitch_loss.detach(),
            "velocity": velocity_loss.detach(),
            "duration": duration_loss.detach(),
            "total": total_loss.detach()
        }

        return total_loss, losses
