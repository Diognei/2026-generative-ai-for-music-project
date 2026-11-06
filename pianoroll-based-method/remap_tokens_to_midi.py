
import os
import sys
import json
from pathlib import Path

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

# Canais MIDI de destino.
# Drums utiliza canal 10 (indice 9).
TRACK_CHANNELS = [
    9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10
]

# Programas GM representantes.
# Drums nao utiliza program_change.
TRACK_PROGRAMS = [
    None, 48, 88, 32, 16, 0, 24, 56, 80, 112, 96
]

NUM_TRACKS = 11
MIN_PITCH = 36
MAX_PITCH = 95
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1

RESOLUTION = 32

FEATURES_PER_GROUP = NUM_PITCHES
PITCH_SIZE = NUM_TRACKS * FEATURES_PER_GROUP
VELOCITY_SIZE = PITCH_SIZE
DURATION_SIZE = PITCH_SIZE

FEATURES_PER_TOKEN = (
    PITCH_SIZE + VELOCITY_SIZE + DURATION_SIZE
)


# ============================================================
# CONVERSAO DE UM NPZ PARA MIDI
# ============================================================

def tokens_to_midi(input_path, output_path):

    data = np.load(input_path, allow_pickle=False)

    tokens = data["tokens"]

    metadata = json.loads(
        data["metadata"].item()
    )

    # ========================================================
    # VALIDACAO
    # ========================================================

    if tokens.ndim != 2:
        raise ValueError(
            f"Esperada matriz 2D, encontrada: {tokens.shape}"
        )

    if tokens.shape[1] != FEATURES_PER_TOKEN:
        raise ValueError(
            f"Esperados {FEATURES_PER_TOKEN} valores por token, "
            f"encontrados {tokens.shape[1]}."
        )

    if metadata["num_tracks"] != NUM_TRACKS:
        raise ValueError("Quantidade de pistas incompatível.")

    if metadata["min_pitch"] != MIN_PITCH:
        raise ValueError("MIN_PITCH incompatível.")

    if metadata["max_pitch"] != MAX_PITCH:
        raise ValueError("MAX_PITCH incompatível.")

    if metadata["resolution"] != RESOLUTION:
        raise ValueError("Resolução temporal incompatível.")

    sequence_length = tokens.shape[0]

    # Recupera a resolucao MIDI original.
    ticks_per_beat = int(metadata["ticks_per_beat"])

    def step_to_tick(step):
        return round(step * ticks_per_beat / RESOLUTION)

    # ========================================================
    # SEPARACAO DOS ATRIBUTOS
    # ========================================================

    pitch_tokens = tokens[:, 0:PITCH_SIZE]

    velocity_tokens = tokens[
        :, PITCH_SIZE:PITCH_SIZE + VELOCITY_SIZE
    ]

    duration_tokens = tokens[
        :, PITCH_SIZE + VELOCITY_SIZE:
    ]

    # ========================================================
    # CRIACAO DO MIDI
    # ========================================================

    midi = mido.MidiFile(
        type=1,
        ticks_per_beat=ticks_per_beat
    )

    # ========================================================
    # RECONSTRUCAO DAS 11 PISTAS
    # ========================================================

    for track_index, track_name in enumerate(TRACK_NAMES):

        channel = TRACK_CHANNELS[track_index]
        program = TRACK_PROGRAMS[track_index]

        track = mido.MidiTrack()
        midi.tracks.append(track)

        track.append(
            mido.MetaMessage(
                "track_name",
                name=track_name,
                time=0
            )
        )

        # Programa GM representativo.
        if program is not None:

            track.append(
                mido.Message(
                    "program_change",
                    channel=channel,
                    program=program,
                    time=0
                )
            )

        events = []
        event_order = 0

        # ====================================================
        # RECUPERACAO DAS NOTAS
        # ====================================================

        for step in range(sequence_length):

            for pitch_index in range(NUM_PITCHES):

                index = (
                    track_index * NUM_PITCHES + pitch_index
                )

                # Verifica se existe inicio de nota.
                if pitch_tokens[step, index] == 0:
                    continue

                velocity = int(
                    velocity_tokens[step, index]
                )

                duration = int(
                    duration_tokens[step, index]
                )

                # Protecao contra valores invalidos.
                velocity = max(1, min(127, velocity))
                duration = max(1, duration)

                pitch = MIN_PITCH + pitch_index

                start_tick = step_to_tick(step)

                end_tick = step_to_tick(
                    step + duration
                )

                # note_on
                events.append(
                    (
                        start_tick,
                        1,
                        event_order,
                        mido.Message(
                            "note_on",
                            channel=channel,
                            note=pitch,
                            velocity=velocity
                        )
                    )
                )

                event_order += 1

                # note_off
                events.append(
                    (
                        end_tick,
                        0,
                        event_order,
                        mido.Message(
                            "note_off",
                            channel=channel,
                            note=pitch,
                            velocity=0
                        )
                    )
                )

                event_order += 1

        # ====================================================
        # ORDENACAO DOS EVENTOS
        # ====================================================

        # Em caso de mesmo instante:
        # 1. note_off
        # 2. note_on
        #
        # Isso permite encerrar uma nota antes de iniciar
        # outra nota da mesma altura.

        events.sort(
            key=lambda event: (
                event[0],
                event[1],
                event[2]
            )
        )

        # ====================================================
        # CONVERSAO PARA DELTA TIME
        # ====================================================

        previous_tick = 0

        for absolute_tick, _, _, msg in events:

            delta_time = absolute_tick - previous_tick

            track.append(
                msg.copy(time=delta_time)
            )

            previous_tick = absolute_tick

        track.append(
            mido.MetaMessage(
                "end_of_track",
                time=0
            )
        )

    # ========================================================
    # SALVAMENTO
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    midi.save(output_path)


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

    npz_files = []

    # Procura arquivos NPZ em todos os subdiretorios.
    for root, _, files in os.walk(input_dir):

        for filename in files:

            if Path(filename).suffix.lower() == ".npz":

                npz_files.append(
                    Path(root) / filename
                )

    processed = 0
    errors = 0

    for input_path in sorted(npz_files):

        relative_path = input_path.relative_to(input_dir)

        output_path = (
            output_dir
            / relative_path.parent
            / (relative_path.stem + ".mid")
        )

        try:

            tokens_to_midi(
                input_path,
                output_path
            )

            processed += 1

            print(
                f"[OK] {relative_path} -> "
                f"{output_path.stat().st_size / 1024:.1f} KB"
            )

        except Exception as e:

            errors += 1

            print(
                f"[ERRO] {relative_path}: {e}"
            )

    print("\n======================================")
    print("PROCESSAMENTO CONCLUIDO")
    print(f"Arquivos encontrados: {len(npz_files)}")
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
            "Uso: python3 remap_tokens_to_midi.py "
            "<input_dir> <output_dir>"
        )

        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_directory(
        input_dir,
        output_dir
    )
