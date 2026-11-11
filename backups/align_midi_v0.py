
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


# ============================================================
# GENERAL UTILITIES
# ============================================================

def get_total_duration(midi):
    return max(
        (inst.get_end_time() for inst in midi.instruments),
        default=0.0
    )


def get_bar_duration(midi, beat_duration):
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]
        beats_per_bar = ts.numerator * 4.0 / ts.denominator
    else:
        beats_per_bar = 4.0

    return beat_duration * beats_per_bar


def get_beat_duration(midi):
    beats = midi.get_beats()

    if len(beats) >= 2:
        intervals = np.diff(beats)
        intervals = intervals[intervals > 0]

        if len(intervals):
            return float(np.median(intervals))

    # Fallback: estimate from drum events.
    drum_notes = [
        note.start
        for inst in midi.instruments
        if inst.is_drum
        for note in inst.notes
    ]

    if len(drum_notes) >= 2:
        intervals = np.diff(sorted(drum_notes))
        intervals = intervals[intervals > 0]

        if len(intervals):
            estimate = float(np.median(intervals))
            return min(max(estimate, 0.25), 1.5)

    return 0.5


def build_beat_grid(midi, duration, drum_inst, beat_duration):
    """
    Builds a beat grid anchored to the first drum event.
    """

    beats = midi.get_beats()

    if len(beats) > 0:
        anchor = float(beats[0])
    else:
        anchor = 0.0

    if drum_inst and drum_inst.notes:
        first_drum = min(n.start for n in drum_inst.notes)

        # Move the beat grid to the nearest drum event.
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

    index = np.argmin(np.abs(grid - time))
    return float(grid[index])


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


def is_string_instrument(inst):
    # General MIDI programs 40-51.
    return 40 <= inst.program <= 51


# ============================================================
# CHORD DETECTION
# ============================================================

def detect_chords(inst, tolerance=CHORD_TOLERANCE):
    """
    Groups notes with approximately equal onset times.

    A chord is a group containing at least MIN_CHORD_NOTES.
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


def identify_string_track(midi):
    candidates = [
        inst for inst in midi.instruments
        if is_string_instrument(inst) and inst.notes
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda inst: (
            len(detect_chords(inst)),
            len(inst.notes)
        )
    )


# ============================================================
# DRUM TRACK RECONSTRUCTION
# ============================================================

def tile_instrument(inst, duration, bar_duration):
    """
    Repeats the instrument's existing pattern across the
    complete music duration.

    The original pattern is preserved, including its rests.
    """

    if not inst.notes:
        return

    source_start = min(n.start for n in inst.notes)
    source_end = max(n.end for n in inst.notes)

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

        for note in inst.notes:
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
# STRING TRACK RECONSTRUCTION
# ============================================================

def synchronize_string_track(inst, grid):
    """
    Synchronizes all notes belonging to the same chord.
    """

    chords = detect_chords(inst)

    if not chords:
        print("[WARNING] No chords detected in string track.")
        return []

    new_notes = []
    synchronized_chords = []

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

        synchronized_chords.append({
            "start": target_time,
            "end": chord["end"] + offset,
            "pitch_classes": chord["pitch_classes"]
        })

    # Preserve notes that were not classified as chords.
    chord_note_ids = {
        id(note)
        for chord in chords
        for note in chord["notes"]
    }

    for note in inst.notes:
        if id(note) not in chord_note_ids:
            new_notes.append(copy.deepcopy(note))

    inst.notes = sorted(
        new_notes,
        key=lambda n: n.start
    )

    return synchronized_chords


# ============================================================
# CHORD-BASED PITCH CORRECTION
# ============================================================

def find_chord_at_time(time, chords, beat_duration):
    """
    Finds the chord active at a given time.

    If no chord is active, finds the nearest chord within
    a limited temporal distance.
    """

    for chord in chords:
        if chord["start"] <= time <= chord["end"]:
            return chord

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
    Moves a pitch to the closest chord tone while preserving
    its original MIDI octave.

    Example:
        Original: 61 (C#)
        Chord:    C, E, G
        Result:   60 (C)

    The note never moves to another octave.
    """

    if not pitch_classes:
        return original_pitch

    octave_base = (original_pitch // 12) * 12

    candidates = [
        octave_base + pc
        for pc in pitch_classes
    ]

    # Keep candidates inside the MIDI pitch range.
    candidates = [
        p for p in candidates
        if 0 <= p <= 127
    ]

    if not candidates:
        return original_pitch

    return min(
        candidates,
        key=lambda p: abs(p - original_pitch)
    )


# ============================================================
# MELODIC TRACK PROCESSING
# ============================================================

def process_melodic_track(
    inst,
    grid,
    chords,
    beat_duration
):
    """
    1. Synchronizes note onsets to the beat grid.
    2. Adjusts pitches according to the active string chord.
    """

    for note in inst.notes:

        # Temporal synchronization.
        target_time = nearest_grid_time(
            note.start,
            grid
        )

        duration = note.end - note.start

        note.start = target_time
        note.end = target_time + duration

        # Chord-based pitch correction.
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
# MAIN PROCESSING
# ============================================================

def fix_midi(input_path):

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    print(f"\nLoading: {input_path}")

    midi = pretty_midi.PrettyMIDI(input_path)

    duration = get_total_duration(midi)

    if duration <= 0:
        raise ValueError("The MIDI file has no valid duration.")

    print(f"Duration: {duration:.2f}s")
    print(f"Instruments: {len(midi.instruments)}")

    # --------------------------------------------------------
    # 1. Identify drum track.
    # --------------------------------------------------------

    drum_inst = identify_drum_track(midi)

    if drum_inst is None:
        raise ValueError(
            "No drum track found. Check the MIDI is_drum flag."
        )

    print(
        f"\n[DRUM] {drum_inst.name or 'Unnamed'} "
        f"({len(drum_inst.notes)} notes)"
    )

    # --------------------------------------------------------
    # 2. Estimate musical timing.
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

    original_drum_notes = len(drum_inst.notes)

    tile_instrument(
        drum_inst,
        duration,
        bar_duration
    )

    print(
        f"[DRUM] Notes: {original_drum_notes} -> "
        f"{len(drum_inst.notes)}"
    )

    # --------------------------------------------------------
    # 4. Identify the string track with most chords.
    # --------------------------------------------------------

    string_inst = identify_string_track(midi)

    chords = []

    if string_inst is not None:

        original_chords = detect_chords(string_inst)

        print(
            f"\n[STRINGS] Selected: "
            f"{string_inst.name or 'Unnamed'}"
        )

        print(
            f"[STRINGS] Detected chords: "
            f"{len(original_chords)}"
        )

        # Synchronize chords with the drum beat grid.
        chords = synchronize_string_track(
            string_inst,
            grid
        )

        # Repeat the string pattern to cover the music.
        tile_instrument(
            string_inst,
            duration,
            bar_duration
        )

        # Recalculate chord groups after repetition.
        chords = detect_chords(string_inst)

        chords = [
            {
                "start": c["start"],
                "end": c["end"],
                "pitch_classes": c["pitch_classes"]
            }
            for c in chords
        ]

        print(
            f"[STRINGS] Final chords: {len(chords)}"
        )

    else:
        print(
            "\n[WARNING] No General MIDI string instrument found. "
            "Pitch correction will be skipped."
        )

    # --------------------------------------------------------
    # 5. Process all remaining melodic tracks.
    # --------------------------------------------------------

    processed = 0

    for inst in midi.instruments:

        if inst is drum_inst:
            continue

        if inst is string_inst:
            continue

        if not inst.notes:
            continue

        print(
            f"\n[TRACK] Processing: "
            f"{inst.name or 'Unnamed'} "
            f"(program={inst.program}, "
            f"notes={len(inst.notes)})"
        )

        process_melodic_track(
            inst,
            grid,
            chords,
            beat_duration
        )

        processed += 1

    # --------------------------------------------------------
    # 6. Save output.
    # --------------------------------------------------------

    base, ext = os.path.splitext(input_path)

    output_path = base + "_fixed.mid"

    midi.write(output_path)

    print("\n===================================")
    print("MIDI PROCESSING COMPLETED")
    print("===================================")
    print(f"Output: {output_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Processed melodic tracks: {processed}")
    print(f"String chords: {len(chords)}")
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
