
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    NUM_TRACKS,
    NUM_PITCHES,
    NUM_OUTPUTS,
    FEATURES_PER_TOKEN,
    MAX_SEQ_LEN
)


class MidiDataset(Dataset):
    def __init__(
        self,
        data_dir,
        max_seq_len=MAX_SEQ_LEN,
        stride=None,
        cache_size=4
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.window_size = max_seq_len + 1
        self.stride = stride or max_seq_len // 2
        self.cache_size = cache_size

        self.files = sorted(self.data_dir.rglob("*.npz"))

        if not self.files:
            raise RuntimeError(
                f"Nenhum arquivo .npz encontrado em: {self.data_dir}"
            )

        self.cache = OrderedDict()
        self.samples = []

        self._build_index()

        if not self.samples:
            raise RuntimeError(
                f"Nenhuma sequência válida encontrada em: {self.data_dir}"
            )

        print(
            f"Dataset: {self.data_dir} | "
            f"Músicas: {len(self.files)} | "
            f"Janelas: {len(self.samples)}"
        )

    def _build_index(self):
        for file_idx, file_path in enumerate(self.files):
            with np.load(file_path, allow_pickle=False) as data:
                if "tokens" not in data:
                    raise ValueError(
                        f"Arquivo sem array 'tokens': {file_path}"
                    )

                tokens = data["tokens"]

                if tokens.ndim != 2:
                    raise ValueError(
                        f"Formato inválido em {file_path}: {tokens.shape}"
                    )

                if tokens.shape[1] != FEATURES_PER_TOKEN:
                    raise ValueError(
                        f"Esperadas {FEATURES_PER_TOKEN} features, "
                        f"encontradas {tokens.shape[1]} em {file_path}"
                    )

                sequence_length = tokens.shape[0]

            if sequence_length < 2:
                continue

            if sequence_length <= self.window_size:
                starts = [0]
            else:
                last_start = sequence_length - self.window_size

                starts = list(
                    range(0, last_start + 1, self.stride)
                )

                if starts[-1] != last_start:
                    starts.append(last_start)

            for start in starts:
                self.samples.append((file_idx, start))

    def _load_file(self, file_idx):
        if file_idx in self.cache:
            self.cache.move_to_end(file_idx)
            return self.cache[file_idx]

        file_path = self.files[file_idx]

        with np.load(file_path, allow_pickle=False) as data:
            tokens = data["tokens"].astype(np.float32)

        self.cache[file_idx] = tokens

        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return tokens

    def _normalize(self, tokens):
        """
        Layout original:
            [pitch (660), velocity (660), duration (660)]

        Normalização:
            pitch:    0 ou 1
            velocity: 0 a 1
            duration: 0 a 1, usando log1p
        """

        pitch = tokens[:, :NUM_OUTPUTS]
        velocity = tokens[:, NUM_OUTPUTS:2 * NUM_OUTPUTS]
        duration = tokens[:, 2 * NUM_OUTPUTS:]

        pitch = (pitch > 0).astype(np.float32)

        velocity = np.clip(
            velocity, 0, 127
        ) / 127.0

        duration = np.clip(
            duration, 0, 65535
        )

        duration = np.log1p(duration) / np.log1p(65535.0)

        return np.concatenate(
            [pitch, velocity, duration],
            axis=1
        ).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_idx, start = self.samples[index]

        tokens = self._load_file(file_idx)

        # Extrai uma janela com um instante adicional
        # para realizar o deslocamento autorregressivo.
        window = tokens[
            start:start + self.window_size
        ]

        actual_length = len(window)

        # Entrada e alvo deslocados em um instante.
        x = window[:-1]
        y = window[1:]

        valid_length = len(x)

        # Preenchimento para manter tamanho fixo.
        if valid_length < self.max_seq_len:
            padding = self.max_seq_len - valid_length

            x = np.pad(
                x,
                ((0, padding), (0, 0)),
                mode="constant"
            )

            y = np.pad(
                y,
                ((0, padding), (0, 0)),
                mode="constant"
            )

        # Normaliza entrada e alvo.
        x = self._normalize(x)
        y = self._normalize(y)

        # Separa os três tipos de alvo.
        pitch_target = y[:, :NUM_OUTPUTS]

        velocity_target = y[
            :, NUM_OUTPUTS:2 * NUM_OUTPUTS
        ]

        duration_target = y[
            :, 2 * NUM_OUTPUTS:
        ]

        valid_mask = torch.zeros(
            self.max_seq_len,
            dtype=torch.bool
        )

        valid_mask[:valid_length] = True

        x = torch.from_numpy(x)

        targets = {
            "pitch": torch.from_numpy(pitch_target),
            "velocity": torch.from_numpy(velocity_target),
            "duration": torch.from_numpy(duration_target)
        }

        return x, targets, valid_mask


def create_datasets():
    train_dataset = MidiDataset(TRAIN_DIR)
    valid_dataset = MidiDataset(VALID_DIR)
    test_dataset = MidiDataset(TEST_DIR)

    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = create_datasets()

    x, targets, mask = train_dataset[0]

    print("\nExemplo de amostra:")
    print("Input:", x.shape)
    print("Pitch target:", targets["pitch"].shape)
    print("Velocity target:", targets["velocity"].shape)
    print("Duration target:", targets["duration"].shape)
    print("Valid mask:", mask.shape)
    print("Instantes válidos:", mask.sum().item())
