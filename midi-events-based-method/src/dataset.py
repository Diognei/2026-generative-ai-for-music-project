from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from midi_events import midi_to_tokens


def collate_batch(batch, pad_id):
    """Preenche janelas curtas para permitir batches de tamanhos variados."""

    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=pad_id
    )
    padded_targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=pad_id
    )
    return padded_inputs, padded_targets


class MIDIDataset(Dataset):
    """Cada arquivo MIDI é convertido em uma sequência independente.

    Sequências longas são divididas em janelas, sem cruzar a fronteira
    entre músicas. Cada item retorna input e target deslocados em 1 token.
    """

    def __init__(self, midi_paths, vocab, block_size=512):
        self.samples = []

        for midi_path in midi_paths:
            tokens = midi_to_tokens(midi_path, vocab)
            if len(tokens) < 2:
                continue

            # Cada janela tem no máximo block_size tokens de entrada.
            for start in range(0, len(tokens) - 1, block_size):
                chunk = tokens[start:start + block_size + 1]
                if len(chunk) >= 2:
                    self.samples.append(torch.tensor(chunk, dtype=torch.long))

        if not self.samples:
            raise ValueError("O dataset não produziu nenhuma amostra.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence = self.samples[index]
        return sequence[:-1], sequence[1:]
