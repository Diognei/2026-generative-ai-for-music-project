
import pretty_midi

TIME_STEP = 0.01
MAX_SHIFT = 100


def midi_to_tokens(midi_path, vocab):
    """Convert MIDI tracks into event sequences."""

    midi = pretty_midi.PrettyMIDI(str(midi_path))

    instruments = [
        instrument
        for instrument in midi.instruments
        if instrument.notes
    ]

    if not instruments:
        raise ValueError(
            f"Nenhuma pista com notas encontrada em {midi_path}"
        )

    tokens = [vocab.encode("MUSIC_START")]
    track_count = 0

    for instrument in instruments:

        tokens.append(vocab.track_start_id)

        if instrument.is_drum:
            tokens.append(vocab.is_drum_id)
        else:
            tokens.append(vocab.is_melodic_id)
            tokens.append(vocab.encode(
                f"PROGRAM_{instrument.program}"
            ))

        events = []

        for note in instrument.notes:
            start_tick = max(0, round(note.start / TIME_STEP))
            end_tick = max(
                start_tick + 1,
                round(note.end / TIME_STEP)
            )

            # NOTE_OFF before NOTE_ON for simultaneous events.
            events.append((end_tick, 0, note.pitch, 0))
            events.append((
                start_tick,
                1,
                note.pitch,
                max(1, min(127, note.velocity))
            ))

        events.sort(key=lambda item: (
            item[0],
            item[1],
            item[2]
        ))

        previous_tick = 0

        for tick, event_type, pitch, velocity in events:

            delta = tick - previous_tick

            while delta > 0:
                shift = min(delta, MAX_SHIFT)

                tokens.append(
                    vocab.encode(f"TIME_SHIFT_{shift}")
                )

                delta -= shift

            previous_tick = tick

            if event_type == 0:
                tokens.append(
                    vocab.encode(f"NOTE_OFF_{pitch}")
                )
            else:
                tokens.append(
                    vocab.encode(f"VELOCITY_{velocity}")
                )
                tokens.append(
                    vocab.encode(f"NOTE_ON_{pitch}")
                )

        tokens.append(vocab.track_end_id)
        track_count += 1

    tokens.append(vocab.music_end_id)

    return tokens


def tokens_to_midi(tokens, vocab, output_path):
    """Reconstruct a MIDI file from events."""

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)

    instruments = []

    current_instrument = None
    current_time = 0.0
    pending_velocity = 80
    active_notes = {}

    for token_id in tokens:

        token = vocab.decode(token_id)

        if token == "TRACK_START":

            current_time = 0.0
            current_instrument = None
            active_notes = {}

        elif token == "IS_DRUM":

            current_instrument = pretty_midi.Instrument(
                program=0,
                is_drum=True
            )

            instruments.append(current_instrument)

        elif token == "IS_MELODIC":

            current_instrument = None

        elif token.startswith("PROGRAM_"):

            program = int(token.split("_")[1])

            current_instrument = pretty_midi.Instrument(
                program=program,
                is_drum=False
            )

            instruments.append(current_instrument)

        elif token.startswith("TIME_SHIFT_"):

            shift = int(token.split("_")[2])
            current_time += shift * TIME_STEP

        elif token.startswith("VELOCITY_"):

            pending_velocity = int(token.split("_")[1])

        elif token.startswith("NOTE_ON_"):

            if current_instrument is not None:

                pitch = int(token.split("_")[2])

                if pitch not in active_notes:
                    active_notes[pitch] = (
                        current_time,
                        pending_velocity
                    )

        elif token.startswith("NOTE_OFF_"):

            if current_instrument is not None:

                pitch = int(token.split("_")[2])

                if pitch in active_notes:

                    start, velocity = active_notes.pop(pitch)

                    end = max(
                        current_time,
                        start + TIME_STEP
                    )

                    current_instrument.notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start,
                            end=end
                        )
                    )

        elif token == "TRACK_END":

            if current_instrument is not None:

                # Close notes that may still be open.
                for pitch, (start, velocity) in active_notes.items():

                    current_instrument.notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start,
                            end=max(
                                current_time,
                                start + TIME_STEP
                            )
                        )
                    )

            active_notes = {}
            current_instrument = None

    # Protect against incomplete sequences.
    if current_instrument is not None:

        for pitch, (start, velocity) in active_notes.items():

            current_instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start,
                    end=max(
                        current_time,
                        start + TIME_STEP
                    )
                )
            )

    if not instruments:
        instruments = [
            pretty_midi.Instrument(program=0)
        ]

    midi.instruments.extend(instruments)

    midi.write(str(output_path))


# The upgraded stream uses one clock for every instrument. These definitions
# intentionally shadow the baseline functions above while keeping this fork
# easy to compare with the original project.
def midi_to_tokens(midi_path, vocab):
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    instruments = [instrument for instrument in midi.instruments if instrument.notes]
    if not instruments:
        raise ValueError(f"Nenhuma pista com notas encontrada em {midi_path}")
    if len(instruments) > vocab.MAX_TRACKS:
        raise ValueError(f"{midi_path} possui mais de {vocab.MAX_TRACKS} pistas.")

    tokens = [vocab.music_start_id]
    for instrument in instruments:
        tokens.append(vocab.track_start_id)
        if instrument.is_drum:
            tokens.append(vocab.is_drum_id)
        else:
            tokens.extend([vocab.is_melodic_id, vocab.encode(f"PROGRAM_{instrument.program}")])
        tokens.append(vocab.track_end_id)

    events = []
    for track_index, instrument in enumerate(instruments):
        for note in instrument.notes:
            start = max(0, round(note.start / TIME_STEP))
            end = max(start + 1, round(note.end / TIME_STEP))
            events.extend([
                (end, 0, track_index, note.pitch, 0),
                (start, 1, track_index, note.pitch, max(1, min(127, note.velocity)))
            ])
    events.sort(key=lambda event: (event[0], event[1], event[2], event[3]))

    previous_tick = 0
    for tick, event_type, track_index, pitch, velocity in events:
        delta = tick - previous_tick
        while delta > 0:
            shift = min(delta, MAX_SHIFT)
            tokens.append(vocab.encode(f"TIME_SHIFT_{shift}"))
            delta -= shift
        previous_tick = tick
        tokens.append(vocab.encode(f"TRACK_SELECT_{track_index}"))
        if event_type == 0:
            tokens.append(vocab.encode(f"NOTE_OFF_{pitch}"))
        else:
            tokens.extend([vocab.encode(f"VELOCITY_{velocity}"), vocab.encode(f"NOTE_ON_{pitch}")])
    tokens.append(vocab.music_end_id)
    return tokens


def tokens_to_midi(tokens, vocab, output_path):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instruments = []
    active_notes = {}
    current_time = 0.0
    current_track = None
    pending_velocity = 80
    index = 1

    while index < len(tokens):
        token = vocab.decode(tokens[index])
        index += 1
        if token == "TRACK_START":
            is_drum = vocab.decode(tokens[index]) == "IS_DRUM"
            index += 1
            program = 0
            if not is_drum:
                program_token = vocab.decode(tokens[index])
                if not program_token.startswith("PROGRAM_"):
                    raise ValueError(
                        f"Programa de pista inválido: {program_token}"
                    )
                program = int(program_token.split("_")[1])
                index += 1
            if vocab.decode(tokens[index]) != "TRACK_END":
                raise ValueError("Cabeçalho de pista inválido.")
            index += 1
            instruments.append(pretty_midi.Instrument(program=program, is_drum=is_drum))
        elif token.startswith("TIME_SHIFT_"):
            current_time += int(token.split("_")[2]) * TIME_STEP
        elif token.startswith("TRACK_SELECT_"):
            current_track = int(token.split("_")[2])
        elif token.startswith("VELOCITY_"):
            pending_velocity = int(token.split("_")[1])
        elif token.startswith("NOTE_ON_") and current_track is not None:
            pitch = int(token.split("_")[2])
            active_notes[(current_track, pitch)] = (current_time, pending_velocity)
        elif token.startswith("NOTE_OFF_") and current_track is not None:
            pitch = int(token.split("_")[2])
            key = (current_track, pitch)
            if key in active_notes:
                start, velocity = active_notes.pop(key)
                instruments[current_track].notes.append(pretty_midi.Note(
                    velocity=velocity, pitch=pitch, start=start,
                    end=max(current_time, start + TIME_STEP)))

    for (track, pitch), (start, velocity) in active_notes.items():
        instruments[track].notes.append(pretty_midi.Note(
            velocity=velocity, pitch=pitch, start=start,
            end=max(current_time, start + TIME_STEP)))
    midi.instruments.extend(instruments)
    if not instruments:
        midi.instruments.append(pretty_midi.Instrument(program=0))
    midi.write(str(output_path))
