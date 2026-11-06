
import pretty_midi

TIME_STEP = 0.01
MAX_SHIFT = 100


def midi_to_tokens(midi_path, vocab):
    """Converte pistas MIDI em sequências de eventos."""

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

            # NOTE_OFF antes de NOTE_ON em eventos simultâneos.
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
    """Reconstrói um arquivo MIDI a partir dos eventos."""

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

                # Fecha notas que eventualmente ficaram abertas.
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

    # Proteção para sequências incompletas.
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
