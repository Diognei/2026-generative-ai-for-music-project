
import os
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import mido


def _load_generate_file_functions():
    generate_files_dir = Path(__file__).resolve().parent / "generate-files"
    sys.path.insert(0, str(generate_files_dir))
    module_functions = {}

    for module_name, function_name in (
        ("generate_pianoroll", "generate_pianoroll"),
        ("generate_videofile", "generate_video"),
        ("generate_wavefile", "midi_to_wav"),
    ):
        module_path = generate_files_dir / f"{module_name}.py"
        module_spec = importlib.util.spec_from_file_location(
            module_name, module_path
        )

        if module_spec is None or module_spec.loader is None:
            raise ImportError(
                f"Não foi possível carregar o módulo: {module_path}"
            )

        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module_functions[function_name] = getattr(module, function_name)

    return module_functions


_GENERATE_FILE_FUNCTIONS = _load_generate_file_functions()
generate_pianoroll = _GENERATE_FILE_FUNCTIONS["generate_pianoroll"]
generate_video = _GENERATE_FILE_FUNCTIONS["generate_video"]
midi_to_wav = _GENERATE_FILE_FUNCTIONS["midi_to_wav"]

from config import (
    DEVICE,
    USE_AMP,
    BEST_MODEL_PATH,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    DIM_FEEDFORWARD,
    MAX_SEQ_LEN,
    DROPOUT,
    NUM_TRACKS,
    NUM_PITCHES,
    NUM_OUTPUTS,
    MIN_PITCH,
    RESOLUTION,
    GENERATION_LENGTH,
    GENERATION_TEMPERATURE,
    PITCH_THRESHOLD,
    TRACK_NAMES
)

from model import MidiTransformer


TRACK_CHANNELS = [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

TRACK_PROGRAMS = [
    None, 48, 88, 32, 16, 0, 24, 56, 80, 112, 96
]

TICKS_PER_BEAT = 480
TEMPO = 500000


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
def generate_sequence(
    model,
    length,
    temperature=GENERATION_TEMPERATURE,
    pitch_threshold=PITCH_THRESHOLD,
    use_amp=False
):
    if temperature <= 0:
        raise ValueError("Temperature deve ser maior que zero.")

    sequence = torch.zeros(
        1, 1, NUM_OUTPUTS * 3,
        device=DEVICE
    )

    generated = []
    total_generated_notes = 0

    print("\nDiagnóstico de probabilidades de pitch:")
    print("-" * 60)

    for step in range(length):

        context = sequence[:, -MAX_SEQ_LEN:, :]

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp
        ):
            predictions = model(context)

        pitch_logits = predictions["pitch_logits"][:, -1, :]
        velocity_pred = predictions["velocity"][:, -1, :]
        duration_pred = predictions["duration"][:, -1, :]

        # Probabilidades originais, antes do threshold.
        probabilities = torch.sigmoid(
            pitch_logits / temperature
        )

        # Diagnóstico das probabilidades.
        if step % 100 == 0:
            print(
                f"Instante {step}: "
                f"max={probabilities.max().item():.6f} | "
                f"mean={probabilities.mean().item():.6f} | "
                f">0.5={(probabilities > 0.5).sum().item()} | "
                f">0.1={(probabilities > 0.1).sum().item()} | "
                f">0.01={(probabilities > 0.01).sum().item()}"
            )

        # Remove probabilidades abaixo do threshold.
        filtered_probabilities = torch.where(
            probabilities >= pitch_threshold,
            probabilities,
            torch.zeros_like(probabilities)
        )

        # Amostragem Bernoulli.
        pitch = torch.bernoulli(filtered_probabilities)

        velocity = velocity_pred.clamp(0.0, 1.0)
        duration = duration_pred.clamp(0.0, 1.0)

        velocity = velocity * pitch
        duration = duration * pitch

        next_token = torch.cat(
            [pitch, velocity, duration],
            dim=-1
        )

        generated.append(next_token.squeeze(0).cpu())

        current_notes = int(pitch.sum().item())
        total_generated_notes += current_notes

        sequence = torch.cat(
            [sequence, next_token.unsqueeze(1)],
            dim=1
        )

        if (step + 1) % 100 == 0:
            print(
                f"Gerados {step + 1}/{length} instantes | "
                f"Notas neste instante: {current_notes} | "
                f"Total: {total_generated_notes}"
            )

    print("-" * 60)
    print(f"Total de notas amostradas: {total_generated_notes}")

    generated = torch.stack(generated, dim=0)

    return generated.numpy()


def sequence_to_midi(sequence, output_path):

    sequence_length = sequence.shape[0]

    pitch_matrix = sequence[:, :NUM_OUTPUTS]
    velocity_matrix = sequence[:, NUM_OUTPUTS:2 * NUM_OUTPUTS]
    duration_matrix = sequence[:, 2 * NUM_OUTPUTS:]

    velocity_matrix = np.clip(
        velocity_matrix * 127.0,
        1,
        127
    )

    duration_matrix = np.expm1(
        np.clip(duration_matrix, 0, 1) * np.log1p(65535.0)
    )

    duration_matrix = np.clip(
        np.rint(duration_matrix),
        1,
        65535
    ).astype(np.int64)

    ticks_per_step = TICKS_PER_BEAT / RESOLUTION

    midi = mido.MidiFile(
        type=1,
        ticks_per_beat=TICKS_PER_BEAT
    )

    total_notes = 0
    track_note_counts = []

    for track_idx in range(NUM_TRACKS):

        midi_track = mido.MidiTrack()
        midi.tracks.append(midi_track)

        channel = TRACK_CHANNELS[track_idx]
        program = TRACK_PROGRAMS[track_idx]

        midi_track.append(
            mido.MetaMessage(
                "track_name",
                name=TRACK_NAMES[track_idx],
                time=0
            )
        )

        if track_idx == 0:
            midi_track.append(
                mido.MetaMessage(
                    "set_tempo",
                    tempo=TEMPO,
                    time=0
                )
            )

            midi_track.append(
                mido.MetaMessage(
                    "time_signature",
                    numerator=4,
                    denominator=4,
                    time=0
                )
            )

        if program is not None:
            midi_track.append(
                mido.Message(
                    "program_change",
                    channel=channel,
                    program=program,
                    time=0
                )
            )

        events = []
        track_notes = 0

        for step in range(sequence_length):

            for pitch_idx in range(NUM_PITCHES):

                output_idx = (
                    track_idx * NUM_PITCHES + pitch_idx
                )

                if pitch_matrix[step, output_idx] < 0.5:
                    continue

                pitch = MIN_PITCH + pitch_idx

                velocity = int(
                    round(velocity_matrix[step, output_idx])
                )

                duration = int(
                    duration_matrix[step, output_idx]
                )

                start_tick = int(round(step * ticks_per_step))

                end_tick = int(
                    round((step + duration) * ticks_per_step)
                )

                velocity = max(1, min(127, velocity))

                events.append((
                    start_tick,
                    1,
                    mido.Message(
                        "note_on",
                        channel=channel,
                        note=pitch,
                        velocity=velocity
                    )
                ))

                events.append((
                    end_tick,
                    0,
                    mido.Message(
                        "note_off",
                        channel=channel,
                        note=pitch,
                        velocity=0
                    )
                ))

                track_notes += 1

        events.sort(key=lambda event: (event[0], event[1]))

        previous_tick = 0

        for absolute_tick, _, message in events:

            delta_tick = absolute_tick - previous_tick

            midi_track.append(
                message.copy(time=delta_tick)
            )

            previous_tick = absolute_tick

        midi_track.append(
            mido.MetaMessage(
                "end_of_track",
                time=0
            )
        )

        total_notes += track_notes
        track_note_counts.append(track_notes)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    midi.save(output_path)

    print("\nNotas por pista:")

    for name, count in zip(TRACK_NAMES, track_note_counts):
        print(f"  {name}: {count}")

    return total_notes


def generate_companion_files(midi_path):

    midi_path = Path(midi_path)

    wav_path = midi_path.with_suffix(".wav")
    png_path = midi_path.with_suffix(".png")
    video_path = midi_path.with_suffix(".mp4")

    midi_to_wav(midi_path, wav_path)
    generate_pianoroll(midi_path, png_path)
    generate_video(wav_path, png_path, video_path)

    return wav_path, png_path, video_path


def main():

    parser = argparse.ArgumentParser(
        description="Geração autorregressiva de música MIDI"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=BEST_MODEL_PATH
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated/generated.mid"
    )

    parser.add_argument(
        "--length",
        type=int,
        default=GENERATION_LENGTH
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=GENERATION_TEMPERATURE
    )

    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=PITCH_THRESHOLD
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    args = parser.parse_args()

    if args.length < 1:
        raise ValueError("O comprimento deve ser maior que zero.")

    if not 0.0 <= args.pitch_threshold <= 1.0:
        raise ValueError("Pitch threshold deve estar entre 0 e 1.")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    use_amp = USE_AMP and DEVICE.type == "cuda"

    print("=" * 60)
    print("MIDI TRANSFORMER GENERATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Generation length: {args.length}")
    print(f"Temperature: {args.temperature}")
    print(f"Pitch threshold: {args.pitch_threshold}")
    print("=" * 60)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {args.checkpoint}"
        )

    model, epoch = load_model(args.checkpoint)

    print(f"\nModelo carregado. Epoch: {epoch}")
    print("\nIniciando geração...\n")

    sequence = generate_sequence(
        model=model,
        length=args.length,
        temperature=args.temperature,
        pitch_threshold=args.pitch_threshold,
        use_amp=use_amp
    )

    print("\nConvertendo sequência para MIDI...")

    total_notes = sequence_to_midi(
        sequence,
        args.output
    )

    print("\n" + "=" * 60)
    print("GERAÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"Arquivo: {args.output}")
    print(f"Instantes gerados: {args.length}")
    print(f"Notas geradas: {total_notes}")

    # 32 passos por semínima e 4 semínimas por compasso.
    bars = args.length / (RESOLUTION * 4)

    # A 120 BPM, cada semínima dura 0,5 segundo.
    seconds = args.length / (RESOLUTION * 2)

    print(f"Duração aproximada: {seconds:.2f} segundos")
    print(f"Compassos aproximados: {bars:.2f}")

    if total_notes == 0:
        print("\nATENÇÃO: nenhuma nota foi gerada.")
        print("Os arquivos complementares não serão produzidos.")
        print("Experimente executar com um pitch threshold menor.")
        print("=" * 60)
        return

    print("\nGerando arquivos complementares...")

    wav_path, png_path, video_path = generate_companion_files(
        args.output
    )

    print(f"Áudio: {wav_path}")
    print(f"Pianoroll: {png_path}")
    print(f"Vídeo: {video_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()