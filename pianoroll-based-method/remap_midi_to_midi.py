
import os
import sys
from pathlib import Path
from collections import defaultdict, deque

import mido


# ============================================================
# CONFIGURACAO
# ============================================================

TRACKS = [
    ("Drums", 9, None),
    ("Ensemble", 0, 48),
    ("Pad", 1, 88),
    ("Bass", 2, 32),
    ("Organ", 3, 16),
    ("Piano", 4, 0),
    ("Guitar", 5, 24),
    ("Brass", 6, 56),
    ("Lead", 7, 80),
    ("Percussive", 8, 112),
    ("Effects", 10, 96),
]

# Mapeamento dos 128 programas GM para as 11 pistas.
# Os programas GM sao numerados de 0 a 127.

PROGRAM_TO_TRACK = {}

def map_programs(programs, track_index):
    for program in programs:
        PROGRAM_TO_TRACK[program] = track_index


map_programs(range(0, 8), 5)       # Piano
map_programs(range(8, 16), 5)      # Chromatic Percussion -> Piano
map_programs(range(16, 24), 4)     # Organ
map_programs(range(24, 32), 6)     # Guitar
map_programs(range(32, 40), 3)     # Bass
map_programs(range(40, 56), 1)     # Strings + Ensemble -> Ensemble
map_programs(range(56, 64), 7)     # Brass
map_programs(range(64, 72), 7)     # Reed -> Brass
map_programs(range(72, 80), 1)     # Pipe -> Ensemble
map_programs(range(80, 88), 8)     # Synth Lead
map_programs(range(88, 96), 2)     # Synth Pad
map_programs(range(96, 104), 10)   # Synth Effects
map_programs(range(104, 112), 6)  # Ethnic -> Guitar
map_programs(range(112, 120), 9)  # Percussive
map_programs(range(120, 128), 10) # Sound Effects -> Effects


# ============================================================
# CONVERSAO DE UM MIDI
# ============================================================

def remap_midi(input_path, output_path):

    midi = mido.MidiFile(input_path)

    # Eventos globais: (tempo absoluto, ordem, mensagem)
    all_events = []
    serial = 0

    for track_index, track in enumerate(midi.tracks):
        absolute_time = 0

        for msg in track:
            absolute_time += msg.time

            if msg.type == "end_of_track":
                continue

            all_events.append(
                (absolute_time, serial, msg.copy())
            )

            serial += 1

    # Ordena todos os eventos cronologicamente.
    all_events.sort(key=lambda x: (x[0], x[1]))

    # Programas ativos em cada canal.
    current_program = defaultdict(int)

    # Fila de grupos para associar note_off ao note_on correspondente.
    active_notes = defaultdict(deque)

    # Eventos separados por pista de destino.
    output_events = [[] for _ in TRACKS]

    # Metadados globais serão colocados na primeira pista.
    metadata_events = []

    for absolute_time, event_order, msg in all_events:

        if msg.is_meta:
            metadata_events.append(
                (absolute_time, event_order, msg)
            )
            continue

        if msg.type == "program_change":
            current_program[msg.channel] = msg.program
            continue

        if msg.type not in ("note_on", "note_off"):
            continue

        channel = msg.channel
        pitch = msg.note

        # Canal 10 (índice 9) é sempre bateria.
        if channel == 9:
            track_index = 0

        else:
            if msg.type == "note_on" and msg.velocity > 0:

                program = current_program[channel]
                track_index = PROGRAM_TO_TRACK[program]

                active_notes[(channel, pitch)].append(track_index)

            else:
                # note_off ou note_on com velocity zero.
                key = (channel, pitch)

                if active_notes[key]:
                    track_index = active_notes[key].popleft()
                else:
                    program = current_program[channel]
                    track_index = PROGRAM_TO_TRACK[program]

        # Remapeia o canal para o canal fixo da pista de destino.
        destination_channel = TRACKS[track_index][1]

        new_msg = msg.copy(channel=destination_channel)

        output_events[track_index].append(
            (absolute_time, event_order, new_msg)
        )

    # ========================================================
    # CRIACAO DO MIDI DE SAIDA
    # ========================================================

    output_midi = mido.MidiFile(
        type=1,
        ticks_per_beat=midi.ticks_per_beat
    )

    for track_index, (track_name, channel, program) in enumerate(TRACKS):

        new_track = mido.MidiTrack()
        output_midi.tracks.append(new_track)

        # Nome da pista.
        new_track.append(
            mido.MetaMessage("track_name", name=track_name, time=0)
        )

        events = output_events[track_index]

        # Metadados globais ficam na primeira pista.
        if track_index == 0:
            events.extend(metadata_events)

        # Programa representativo da pista.
        if program is not None:
            events.append(
                (
                    0,
                    -1,
                    mido.Message(
                        "program_change",
                        channel=channel,
                        program=program,
                        time=0
                    )
                )
            )

        # Ordena eventos pelo tempo absoluto.
        events.sort(key=lambda x: (x[0], x[1]))

        previous_time = 0

        for absolute_time, _, msg in events:

            delta_time = absolute_time - previous_time

            new_track.append(
                msg.copy(time=delta_time)
            )

            previous_time = absolute_time

        new_track.append(
            mido.MetaMessage("end_of_track", time=0)
        )

    # Cria a pasta de destino, se necessário.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_midi.save(output_path)


# ============================================================
# PROCESSAMENTO RECURSIVO DA PASTA
# ============================================================

def process_directory(input_dir, output_dir):

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.is_dir():
        raise ValueError(f"Pasta de entrada inexistente: {input_dir}")

    processed = 0
    errors = 0

    for root, _, files in os.walk(input_dir):

        for filename in files:

            if Path(filename).suffix.lower() not in (".mid", ".midi"):
                continue

            input_path = Path(root) / filename

            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path

            try:
                remap_midi(input_path, output_path)

                processed += 1

                print(f"[OK] {relative_path}")

            except Exception as e:

                errors += 1

                print(f"[ERRO] {relative_path}: {e}")

    print("\n================================")
    print("PROCESSAMENTO CONCLUIDO")
    print(f"MIDIs processados: {processed}")
    print(f"Erros: {errors}")
    print(f"Diretorio de saida: {output_dir}")
    print("================================")


# ============================================================
# EXECUCAO
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Uso: python3 remap_midis.py <input_dir> <output_dir>"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_directory(input_dir, output_dir)
