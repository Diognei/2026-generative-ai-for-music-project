
import os
import sys
import json
from pathlib import Path
from collections import defaultdict, deque

import mido
import numpy as np


# ============================================================
# CONFIGURACAO
# ============================================================

TRACK_NAMES = [
    "Drums",
    "Ensemble",
    "Pad",
    "Bass",
    "Organ",
    "Piano",
    "Guitar",
    "Brass",
    "Lead",
    "Percussive",
    "Effects",
]

NUM_TRACKS = 11

MIN_PITCH = 36
MAX_PITCH = 95
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1

# Cada semínima sera dividida em 32 intervalos.
RESOLUTION = 32

FEATURES_PER_GROUP = NUM_PITCHES * 3
FEATURES_PER_TOKEN = NUM_TRACKS * FEATURES_PER_GROUP

PITCH_SIZE = NUM_TRACKS * NUM_PITCHES
VELOCITY_SIZE = PITCH_SIZE
DURATION_SIZE = PITCH_SIZE


# ============================================================
# LEITURA E CONVERSAO DE UM MIDI
# ============================================================

def midi_to_tokens(input_path):

    midi = mido.MidiFile(input_path)

    if len(midi.tracks) != NUM_TRACKS:
        raise ValueError(
            f"Esperadas {NUM_TRACKS} pistas, "
            f"encontradas {len(midi.tracks)}."
        )

    # Quantidade de ticks correspondente a uma coluna.
    ticks_per_step = midi.ticks_per_beat / RESOLUTION

    # Cada entrada representa:
    # (track, pitch, inicio, velocity, duracao)
    notes = defaultdict(list)

    max_tick = 0
    ignored_pitches = 0
    unmatched_notes = 0

    # ========================================================
    # EXTRACAO DAS NOTAS
    # ========================================================

    for track_index, track in enumerate(midi.tracks):

        absolute_time = 0

        # Fila de notas ativas por canal e pitch.
        active_notes = defaultdict(deque)

        for msg in track:

            absolute_time += msg.time
            max_tick = max(max_tick, absolute_time)

            if msg.is_meta:
                continue

            if msg.type == "note_on" and msg.velocity > 0:

                active_notes[(msg.channel, msg.note)].append(
                    (absolute_time, msg.velocity)
                )

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):

                key = (msg.channel, msg.note)

                if not active_notes[key]:
                    unmatched_notes += 1
                    continue

                start_tick, velocity = active_notes[key].popleft()

                # Verifica se o pitch pertence ao intervalo escolhido.
                if not MIN_PITCH <= msg.note <= MAX_PITCH:
                    ignored_pitches += 1
                    continue

                # Quantiza inicio e fim para a resolucao temporal.
                start_step = round(start_tick / ticks_per_step)
                end_step = round(absolute_time / ticks_per_step)

                # Garante duracao minima de um intervalo.
                duration = max(1, end_step - start_step)

                notes[
                    (track_index, msg.note - MIN_PITCH, start_step)
                ].append((velocity, duration))

        # Notas sem note_off correspondente são ignoradas.
        for queue in active_notes.values():
            unmatched_notes += len(queue)

    # ========================================================
    # COMPRIMENTO DA SEQUENCIA
    # ========================================================

    sequence_length = int(np.ceil(max_tick / ticks_per_step))

    # Garante espaço para a última nota, mesmo se o MIDI
    # possuir um end_of_track anterior ao encerramento dela.
    for (track, pitch, start), repetitions in notes.items():

        for velocity, duration in repetitions:
            sequence_length = max(
                sequence_length,
                start + duration
            )

    # ========================================================
    # MATRIZES DE ATRIBUTOS
    # ========================================================

    # Dimensao:
    # (tempo, instrumentos, pitches)

    pitch_matrix = np.zeros(
        (sequence_length, NUM_TRACKS, NUM_PITCHES),
        dtype=np.uint8
    )

    velocity_matrix = np.zeros(
        (sequence_length, NUM_TRACKS, NUM_PITCHES),
        dtype=np.uint8
    )

    duration_matrix = np.zeros(
        (sequence_length, NUM_TRACKS, NUM_PITCHES),
        dtype=np.uint16
    )

    # ========================================================
    # CONSOLIDACAO DE NOTAS REPETIDAS
    # ========================================================

    for (track, pitch, start), repetitions in notes.items():

        velocities = [item[0] for item in repetitions]
        durations = [item[1] for item in repetitions]

        # Uma unica nota para cada combinacao:
        # instrumento + pitch + instante.

        mean_velocity = round(np.mean(velocities))
        mean_duration = round(np.mean(durations))

        if mean_duration > np.iinfo(np.uint16).max:
            raise ValueError(
                "Duracao excede o limite de uint16."
            )

        pitch_matrix[start, track, pitch] = 1

        velocity_matrix[start, track, pitch] = mean_velocity

        duration_matrix[start, track, pitch] = mean_duration

    # ========================================================
    # CONSTRUCAO DOS TOKENS
    # ========================================================

    # Cada coluna e organizada em tres blocos:
    #
    # [660 pitches | 660 velocities | 660 durations]
    #
    # A ordem dos instrumentos dentro de cada bloco
    # segue TRACK_NAMES.

    pitch_tokens = pitch_matrix.reshape(
        sequence_length, PITCH_SIZE
    )

    velocity_tokens = velocity_matrix.reshape(
        sequence_length, VELOCITY_SIZE
    )

    duration_tokens = duration_matrix.reshape(
        sequence_length, DURATION_SIZE
    )

    tokens = np.concatenate(
        [
            pitch_tokens,
            velocity_tokens,
            duration_tokens
        ],
        axis=1
    )

    # ========================================================
    # METADADOS
    # ========================================================

    metadata = {
        "source_file": Path(input_path).name,
        "ticks_per_beat": midi.ticks_per_beat,
        "resolution": RESOLUTION,
        "ticks_per_step": ticks_per_step,
        "num_tracks": NUM_TRACKS,
        "track_names": TRACK_NAMES,
        "min_pitch": MIN_PITCH,
        "max_pitch": MAX_PITCH,
        "num_pitches": NUM_PITCHES,
        "features_per_token": FEATURES_PER_TOKEN,
        "sequence_length": sequence_length,
        "pitch_values": "0 or 1",
        "velocity_values": "0 to 127",
        "duration_unit": "1/32 quarter note",
        "ignored_pitches": ignored_pitches,
        "unmatched_notes": unmatched_notes
    }

    return tokens, metadata


# ============================================================
# PROCESSAMENTO RECURSIVO
# ============================================================

def process_directory(input_dir, output_dir):

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.is_dir():
        raise ValueError(
            f"Diretorio inexistente: {input_dir}"
        )

    midi_files = []

    # Localiza todos os MIDIs, incluindo subdiretorios.
    for root, _, files in os.walk(input_dir):

        for filename in files:

            if Path(filename).suffix.lower() in (".mid", ".midi"):

                midi_files.append(
                    Path(root) / filename
                )

    processed = 0
    errors = 0

    for input_path in sorted(midi_files):

        relative_path = input_path.relative_to(input_dir)

        # Mantem a estrutura de diretorios e o nome original.
        output_path = (
            output_dir / relative_path.parent /
            (relative_path.stem + ".npz")
        )

        try:

            tokens, metadata = midi_to_tokens(input_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            np.savez_compressed(
                output_path,
                tokens=tokens,
                metadata=json.dumps(metadata)
            )

            processed += 1

            print(
                f"[OK] {relative_path} "
                f"-> {tokens.shape} "
                f"({output_path.stat().st_size / 1024:.1f} KB)"
            )

        except Exception as e:

            errors += 1

            print(f"[ERRO] {relative_path}: {e}")

    print("\n======================================")
    print("PROCESSAMENTO CONCLUIDO")
    print(f"Arquivos encontrados: {len(midi_files)}")
    print(f"Processados: {processed}")
    print(f"Erros: {errors}")
    print(f"Diretorio de saida: {output_dir}")
    print("======================================")


# ============================================================
# EXECUCAO
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Uso: python3 remap_midi_to_tokens.py "
            "<input_dir> <output_dir>"
        )

        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_directory(input_dir, output_dir)
