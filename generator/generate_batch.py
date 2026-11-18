"""Gera N músicas para avaliação nas mesmas condições de generate.py.

Reaproveita o próprio generate.py: mesma função generate(), mesmas
constantes de amostragem (TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY,
MAX_NEW_TOKENS, MAX_TRACKS...), mesmo carregamento do checkpoint e mesma
conversão MIDI -> WAV (midi_to_wav com o SoundFont do projeto e
normalização de loudness). A única diferença é não gerar pianoroll, mp3 e
vídeo, e fixar uma seed por música para o conjunto ser reproduzível.

Para cada música i são salvos <output_dir>/song_XXX.mid e .wav (versão
sem alinhamento, como definido para os conjuntos de avaliação).

Uso:
    python3 generator/generate_batch.py output/lstm_model_epoch_190.pt \
        output/evaluation/set3_lstm_results --n 100
"""

from pathlib import Path
import argparse
import random

import numpy as np
import torch

import generate as gen
from vocabulary import Vocabulary
from model import build_model
from midi_events import tokens_to_midi
from generate_wavefile import midi_to_wav


def load_model(checkpoint_path, device):
    """Mesmo carregamento de generate.main()."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    vocab = Vocabulary()
    if checkpoint["vocab_size"] != vocab.size:
        raise RuntimeError("O checkpoint usa um vocabulário incompatível.")

    # Checkpoints antigos não têm "arch"/"model_config": são do Transformer.
    arch = checkpoint.get("arch", "transformer")
    model_config = checkpoint.get("model_config") or {
        key: checkpoint[key]
        for key in ("block_size", "d_model", "n_heads", "n_layers", "dropout")
    }
    model = build_model(arch, checkpoint["vocab_size"], model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Checkpoint inválido: {name}")

    return model, vocab, arch, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Gera N músicas (MIDI + WAV) a partir de um checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n", type=int, default=100, help="Número de músicas (padrão: 100).")
    parser.add_argument("--seed", type=int, default=0, help="Seed base; a música i usa seed+i.")
    parser.add_argument("--no-wav", action="store_true", help="Salva só os MIDIs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, arch, checkpoint = load_model(args.checkpoint, device)
    print(
        f"Checkpoint: {args.checkpoint} | arch={arch} | epoch={checkpoint.get('epoch')} "
        f"| val={checkpoint.get('validation_loss')} | device={device}",
        flush=True,
    )
    print(
        f"TEMPERATURE={gen.TEMPERATURE} TOP_K={gen.TOP_K} TOP_P={gen.TOP_P} "
        f"REPETITION_PENALTY={gen.REPETITION_PENALTY} REPETITION_WINDOW={gen.REPETITION_WINDOW} "
        f"MAX_NEW_TOKENS={gen.MAX_NEW_TOKENS} MAX_TRACKS={gen.MAX_TRACKS} "
        f"USE_REF_FILE={gen.USE_REF_FILE}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.n):
        midi_path = args.output_dir / f"song_{index:03d}.mid"
        wav_path = midi_path.with_suffix(".wav")

        if not midi_path.exists():
            seed = args.seed + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            tokens = gen.generate(
                model,
                vocab,
                device,
                temperature=gen.TEMPERATURE,
                max_new_tokens=gen.MAX_NEW_TOKENS,
            )
            tokens_to_midi(tokens, vocab, midi_path)
            print(f"[{index + 1}/{args.n}] {midi_path} ({len(tokens)} tokens, seed={seed})", flush=True)
        else:
            print(f"[{index + 1}/{args.n}] já existe: {midi_path}", flush=True)

        if not args.no_wav and not wav_path.exists():
            midi_to_wav(midi_path, wav_path, soundfont=gen.SOUNDFONT_PATH)


if __name__ == "__main__":
    main()
