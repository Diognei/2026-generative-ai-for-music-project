
import argparse
import copy
import math
import os
import pretty_midi
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CHORD_TOLERANCE = 0.12
BEAT_SUBDIVISIONS = 2
MIN_CHORD_NOTES = 2
SUSTAIN_PEDAL_CONTROLLER = 64
SUSTAIN_PEDAL_ON = 127
SUSTAIN_PEDAL_OFF = 0
GUITAR_PROGRAM_RANGE = range(24, 32)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def get_total_duration(midi):
    return max(
        (inst.get_end_time() for inst in midi.instruments),
        default=0.0
    )


def get_beat_duration(midi):
    beats = midi.get_beats()

    if len(beats) >= 2:
        intervals = np.diff(beats)
        intervals = intervals[intervals > 0]

        if len(intervals):
            return float(np.median(intervals))

    drum_times = sorted(
        n.start
        for inst in midi.instruments
        if inst.is_drum
        for n in inst.notes
    )

    if len(drum_times) >= 2:
        intervals = np.diff(drum_times)
        intervals = intervals[intervals > 0]

        if len(intervals):
            return min(max(float(np.median(intervals)), 0.25), 1.5)

    return 0.5


def get_bar_duration(midi, beat_duration):
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]
        beats_per_bar = ts.numerator * 4.0 / ts.denominator
    else:
        beats_per_bar = 4.0

    return beat_duration * beats_per_bar


def is_guitar_instrument(inst):
    """Retorna se a pista usa um programa ou nome de guitarra."""

    if inst.program in GUITAR_PROGRAM_RANGE:
        return True

    name = (inst.name or "").lower()
    return "guitar" in name or "violão" in name or "violao" in name


def get_measure_boundaries(midi, duration, beat_duration):
    """Calcula o início de cada compasso e o fim do último compasso."""

    if midi.time_signature_changes:
        time_signature = midi.time_signature_changes[0]
        beats_per_bar = (
            time_signature.numerator * 4.0 / time_signature.denominator
        )
        anchor = time_signature.time
    else:
        beats_per_bar = 4.0
        beats = midi.get_beats()
        anchor = float(beats[0]) if len(beats) else 0.0

    bar_duration = beat_duration * beats_per_bar
    boundaries = [max(0.0, anchor)]

    while boundaries[-1] < duration:
        boundaries.append(boundaries[-1] + bar_duration)

    return boundaries


def add_guitar_sustain_pedal(midi, duration, beat_duration):
    """Liga o sustain no início de cada compasso e solta no fim dele."""

    boundaries = get_measure_boundaries(
        midi,
        duration,
        beat_duration,
    )

    guitar_tracks = [
        instrument
        for instrument in midi.instruments
        if is_guitar_instrument(instrument)
    ]

    for instrument in guitar_tracks:
        instrument.control_changes = [
            control
            for control in instrument.control_changes
            if control.number != SUSTAIN_PEDAL_CONTROLLER
        ]

        pedal_events = []
        for start, end in zip(boundaries, boundaries[1:]):
            if start >= duration:
                continue

            pedal_events.append(pretty_midi.ControlChange(
                number=SUSTAIN_PEDAL_CONTROLLER,
                value=SUSTAIN_PEDAL_ON,
                time=start,
            ))
            pedal_events.append(pretty_midi.ControlChange(
                number=SUSTAIN_PEDAL_CONTROLLER,
                value=SUSTAIN_PEDAL_OFF,
                time=min(end, duration),
            ))

        instrument.control_changes.extend(pedal_events)
        instrument.control_changes.sort(key=lambda control: control.time)

        print(
            f"[PEDAL] {instrument.name or 'Unnamed'} | "
            f"Program: {instrument.program} | "
            f"Measures: {max(0, len(boundaries) - 1)}"
        )

    return len(guitar_tracks)


def build_beat_grid(midi, duration, drum_inst, beat_duration):
    beats = midi.get_beats()

    anchor = float(beats[0]) if len(beats) else 0.0

    if drum_inst and drum_inst.notes:
        first_drum = min(n.start for n in drum_inst.notes)

        anchor += round(
            (first_drum - anchor) / beat_duration
        ) * beat_duration

    step = beat_duration / BEAT_SUBDIVISIONS

    start_index = math.floor(-anchor / step) - 2
    end_index = math.ceil((duration - anchor) / step) + 2

    return np.array([
        anchor + i * step
        for i in range(start_index, end_index + 1)
    ])


def nearest_grid_time(time, grid):
    if len(grid) == 0:
        return time

    return float(grid[np.argmin(np.abs(grid - time))])


def copy_note(note, start, end):
    return pretty_midi.Note(
        velocity=note.velocity,
        pitch=note.pitch,
        start=max(0.0, start),
        end=max(0.001, end)
    )


# ============================================================
# INSTRUMENT IDENTIFICATION
# ============================================================

def identify_drum_track(midi):
    drums = [
        inst for inst in midi.instruments
        if inst.is_drum and inst.notes
    ]

    if not drums:
        return None

    return max(drums, key=lambda inst: len(inst.notes))


# ============================================================
# CHORD DETECTION
# ============================================================

def detect_chords(inst, tolerance=CHORD_TOLERANCE):
    """
    Identifies groups of notes with approximately equal
    onset times.

    Each group with at least MIN_CHORD_NOTES is considered
    a chord.
    """

    notes = sorted(inst.notes, key=lambda n: n.start)

    groups = []
    i = 0

    while i < len(notes):
        reference_time = notes[i].start
        group = []

        while (
            i < len(notes)
            and notes[i].start - reference_time <= tolerance
        ):
            group.append(notes[i])
            i += 1

        if len(group) >= MIN_CHORD_NOTES:
            groups.append({
                "start": min(n.start for n in group),
                "end": max(n.end for n in group),
                "notes": group,
                "pitch_classes": sorted({
                    n.pitch % 12 for n in group
                })
            })

    return groups


def identify_chord_track(midi, drum_inst):
    """
    Selects the instrument with the greatest number of
    detected chords, regardless of its instrument program.

    Drum tracks are excluded.
    """

    candidates = [
        inst for inst in midi.instruments
        if inst is not drum_inst
        and not inst.is_drum
        and inst.notes
    ]

    if not candidates:
        return None

    results = []

    for inst in candidates:
        chords = detect_chords(inst)

        results.append({
            "instrument": inst,
            "chords": chords,
            "count": len(chords)
        })

        print(
            f"[CHORD DETECTION] "
            f"{inst.name or 'Unnamed'} | "
            f"Program: {inst.program} | "
            f"Chords: {len(chords)}"
        )

    # Select the instrument with the most detected chords.
    # The number of notes is used only as a tie-breaker.
    best = max(
        results,
        key=lambda x: (
            x["count"],
            len(x["instrument"].notes)
        )
    )

    if best["count"] == 0:
        return None

    return best["instrument"]


# ============================================================
# PATTERN REPETITION
# ============================================================

def tile_instrument(inst, duration, bar_duration):
    """
    Repeats the existing instrument pattern across the
    complete music duration.

    Preserves the original notes and their relative timing.
    """

    if not inst.notes:
        return

    source_notes = copy.deepcopy(inst.notes)

    source_start = min(n.start for n in source_notes)
    source_end = max(n.end for n in source_notes)

    pattern_duration = source_end - source_start

    if pattern_duration <= 0:
        return

    cycle_duration = (
        math.ceil(pattern_duration / bar_duration)
        * bar_duration
    )

    cycle_duration = max(cycle_duration, bar_duration)

    new_notes = []

    first_cycle = math.floor(
        (0.0 - source_end) / cycle_duration
    ) - 1

    last_cycle = math.ceil(
        (duration - source_start) / cycle_duration
    ) + 1

    for cycle in range(first_cycle, last_cycle + 1):
        offset = cycle * cycle_duration

        for note in source_notes:
            start = note.start + offset
            end = note.end + offset

            if start >= duration or end <= 0:
                continue

            start = max(0.0, start)
            end = min(duration, end)

            if end <= start:
                continue

            new_notes.append(
                copy_note(note, start, end)
            )

    inst.notes = sorted(
        new_notes,
        key=lambda n: n.start
    )


# ============================================================
# DRUM TRACK RECONSTRUCTION
# ============================================================

def reconstruct_drum_track(inst, duration, bar_duration):
    original_count = len(inst.notes)

    tile_instrument(
        inst,
        duration,
        bar_duration
    )

    print(
        f"[DRUM] Notes: {original_count} -> "
        f"{len(inst.notes)}"
    )


# ============================================================
# CHORD TRACK SYNCHRONIZATION
# ============================================================

def synchronize_chord_track(inst, grid):
    """
    Synchronizes all notes belonging to the same chord
    without changing their pitches or instrument identity.
    """

    chords = detect_chords(inst)

    if not chords:
        return

    new_notes = []

    chord_note_ids = {
        id(note)
        for chord in chords
        for note in chord["notes"]
    }

    for chord in chords:
        target_time = nearest_grid_time(
            chord["start"],
            grid
        )

        offset = target_time - chord["start"]

        for note in chord["notes"]:
            new_notes.append(
                copy_note(
                    note,
                    note.start + offset,
                    note.end + offset
                )
            )

    # Preserve notes not classified as chords.
    for note in inst.notes:
        if id(note) not in chord_note_ids:
            new_notes.append(copy.deepcopy(note))

    inst.notes = sorted(
        new_notes,
        key=lambda n: n.start
    )


def prepare_chord_reference(inst, grid, duration, bar_duration):
    """
    Synchronizes the selected chord track and repeats
    its pattern to cover the entire music.
    """

    synchronize_chord_track(inst, grid)

    tile_instrument(
        inst,
        duration,
        bar_duration
    )

    chords = detect_chords(inst)

    reference = [
        {
            "start": chord["start"],
            "end": chord["end"],
            "pitch_classes": chord["pitch_classes"]
        }
        for chord in chords
    ]

    return reference


# ============================================================
# CHORD-BASED PITCH CORRECTION
# ============================================================

def find_chord_at_time(time, chords, beat_duration):
    """
    Finds the chord active at the specified time.

    If no chord is active, searches for the nearest chord
    within a limited temporal distance.
    """

    active = [
        chord for chord in chords
        if chord["start"] <= time <= chord["end"]
    ]

    if active:
        return min(
            active,
            key=lambda c: abs(c["start"] - time)
        )

    if not chords:
        return None

    nearest = min(
        chords,
        key=lambda c: abs(c["start"] - time)
    )

    if abs(nearest["start"] - time) <= beat_duration * 1.5:
        return nearest

    return None


def autotune_pitch(original_pitch, pitch_classes):
    """
    Adjusts the pitch to the closest chord tone,
    preserving the original octave.
    """

    if not pitch_classes:
        return original_pitch

    octave_base = (original_pitch // 12) * 12

    candidates = [
        octave_base + pc
        for pc in pitch_classes
        if 0 <= octave_base + pc <= 127
    ]

    if not candidates:
        return original_pitch

    return min(
        candidates,
        key=lambda p: abs(p - original_pitch)
    )


# ============================================================
# OTHER INSTRUMENT TRACKS
# ============================================================

def process_melodic_track(
    inst,
    grid,
    chords,
    beat_duration
):
    """
    Synchronizes note onsets with the drum grid and
    adjusts pitches according to the chord reference.
    """

    for note in inst.notes:

        # 1. Synchronize note timing.
        target_time = nearest_grid_time(
            note.start,
            grid
        )

        original_duration = note.end - note.start

        note.start = target_time
        note.end = target_time + original_duration

        # 2. Adjust pitch according to the chord.
        chord = find_chord_at_time(
            note.start,
            chords,
            beat_duration
        )

        if chord is not None:
            note.pitch = autotune_pitch(
                note.pitch,
                chord["pitch_classes"]
            )

    inst.notes.sort(key=lambda n: n.start)


# ============================================================
# MAIN
# ============================================================

def fix_midi(input_path):

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    print(f"\nLoading: {input_path}")

    midi = pretty_midi.PrettyMIDI(input_path)

    duration = get_total_duration(midi)

    if duration <= 0:
        raise ValueError("MIDI has no valid duration.")

    print(f"Duration: {duration:.2f}s")
    print(f"Instruments: {len(midi.instruments)}")

    # --------------------------------------------------------
    # 1. Identify drum track.
    # --------------------------------------------------------

    drum_inst = identify_drum_track(midi)

    if drum_inst is None:
        raise ValueError("No drum track found.")

    print(
        f"\n[DRUM] Selected: "
        f"{drum_inst.name or 'Unnamed'}"
    )

    # --------------------------------------------------------
    # 2. Calculate musical timing.
    # --------------------------------------------------------

    beat_duration = get_beat_duration(midi)

    bar_duration = get_bar_duration(
        midi,
        beat_duration
    )

    print(f"[TIMING] Beat: {beat_duration:.3f}s")
    print(f"[TIMING] Bar: {bar_duration:.3f}s")

    grid = build_beat_grid(
        midi,
        duration,
        drum_inst,
        beat_duration
    )

    # --------------------------------------------------------
    # 3. Reconstruct drum track.
    # --------------------------------------------------------

    reconstruct_drum_track(
        drum_inst,
        duration,
        bar_duration
    )

    # --------------------------------------------------------
    # 4. Identify chord track.
    # --------------------------------------------------------

    chord_inst = identify_chord_track(
        midi,
        drum_inst
    )

    chords = []

    if chord_inst is not None:

        print(
            f"\n[CHORD TRACK] Selected: "
            f"{chord_inst.name or 'Unnamed'}"
        )

        print(
            f"[CHORD TRACK] Program: "
            f"{chord_inst.program}"
        )

        print(
            f"[CHORD TRACK] Original notes: "
            f"{len(chord_inst.notes)}"
        )

        # Synchronize and reconstruct chord track.
        chords = prepare_chord_reference(
            chord_inst,
            grid,
            duration,
            bar_duration
        )

        print(
            f"[CHORD TRACK] Final chords: "
            f"{len(chords)}"
        )

    else:
        print(
            "\n[WARNING] No chord track detected. "
            "Pitch correction will be skipped."
        )

    # --------------------------------------------------------
    # 5. Process all remaining instruments.
    # --------------------------------------------------------

    processed = 0

    for inst in midi.instruments:

        if inst is drum_inst:
            continue

        if inst is chord_inst:
            continue

        if not inst.notes:
            continue

        print(
            f"\n[TRACK] Processing: "
            f"{inst.name or 'Unnamed'} | "
            f"Program: {inst.program} | "
            f"Notes: {len(inst.notes)}"
        )

        process_melodic_track(
            inst,
            grid,
            chords,
            beat_duration
        )

        processed += 1

    guitar_tracks = add_guitar_sustain_pedal(
        midi,
        duration,
        beat_duration,
    )

    # --------------------------------------------------------
    # 6. Save MIDI.
    # --------------------------------------------------------

    base, ext = os.path.splitext(input_path)

    output_path = base + "_fixed.mid"

    midi.write(output_path)

    print("\n===================================")
    print("MIDI PROCESSING COMPLETED")
    print("===================================")
    print(f"Output: {output_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Processed tracks: {processed}")
    print(f"Chord reference: {chord_inst.name if chord_inst else 'None'}")
    print(f"Detected chords: {len(chords)}")
    print(f"Guitar tracks with sustain pedal: {guitar_tracks}")
    print("===================================\n")

    return output_path


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Synchronize and correct Transformer-generated MIDI."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input MIDI file."
    )

    args = parser.parse_args()

    fix_midi(args.input)
