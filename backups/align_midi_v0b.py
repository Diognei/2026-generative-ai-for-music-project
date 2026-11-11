"""Align melodic MIDI tracks to a drum reference and chord harmony.

The original MIDI is kept untouched. The corrected file is written next to it
with the suffix ``_fixed.mid``.
"""

import argparse
import copy
import math
import os
import statistics

import pretty_midi


# Timing is quantized to the beat and to the drum onset grid.
BEAT_SUBDIVISIONS = 2
DRUM_ONSET_TOLERANCE = 0.015
CHORD_ONSET_TOLERANCE = 0.10
MIN_CHORD_NOTES = 2
MAX_CHORD_LOOKAHEAD_BEATS = 2.0
MIN_NOTE_DURATION = 0.001


def total_duration(midi):
    return max(
        (instrument.get_end_time() for instrument in midi.instruments),
        default=0.0,
    )


def unique_times(times, tolerance):
    """Return sorted onset times with near-duplicates collapsed."""

    result = []
    for value in sorted(float(time) for time in times):
        if not result or value - result[-1] > tolerance:
            result.append(value)
    return result


def drum_onsets(drum_track):
    return unique_times(
        (note.start for note in drum_track.notes),
        DRUM_ONSET_TOLERANCE,
    )


def estimate_beat_duration(midi, drum_track):
    """Estimate one beat, preferring the MIDI tempo map over drum gaps."""

    beats = [float(beat) for beat in midi.get_beats()]
    beat_intervals = [
        right - left
        for left, right in zip(beats, beats[1:])
        if right - left > 0.001
    ]

    if beat_intervals:
        return statistics.median(beat_intervals)

    onsets = drum_onsets(drum_track)
    onset_intervals = [
        right - left
        for left, right in zip(onsets, onsets[1:])
        if right - left > 0.05
    ]

    if onset_intervals:
        # Drum tracks often contain eighth/sixteenth notes. The largest
        # common gap is a better beat estimate than the smallest gap.
        candidate = statistics.median(onset_intervals)
        while candidate < 0.25:
            candidate *= 2
        return min(max(candidate, 0.25), 1.5)

    return 0.5


def build_reference_grid(midi, drum_track, duration, beat_duration):
    """Build a quantization grid anchored by actual drum attacks.

    The regular beat grid handles rests and weak drum beats. Drum onset times
    are then added as hard temporal anchors, so a melodic attack can land on
    the same timestamp as a real percussion event.
    """

    onsets = drum_onsets(drum_track)
    anchor = onsets[0] if onsets else 0.0
    step = beat_duration / BEAT_SUBDIVISIONS

    regular_grid = []
    first_index = math.floor((0.0 - anchor) / step) - 2
    last_index = math.ceil((duration - anchor) / step) + 2

    for index in range(first_index, last_index + 1):
        time = anchor + index * step
        if -step <= time <= duration + step:
            regular_grid.append(time)

    return unique_times(regular_grid + onsets, DRUM_ONSET_TOLERANCE)


def nearest_grid_time(time, grid):
    if not grid:
        return max(0.0, time)
    return min(grid, key=lambda grid_time: abs(grid_time - time))


def copy_note(note, start, end, pitch=None):
    return pretty_midi.Note(
        velocity=note.velocity,
        pitch=note.pitch if pitch is None else pitch,
        start=max(0.0, start),
        end=max(max(0.0, start) + MIN_NOTE_DURATION, end),
    )


def identify_drum_track(midi):
    candidates = [
        instrument
        for instrument in midi.instruments
        if instrument.is_drum and instrument.notes
    ]
    return max(candidates, key=lambda instrument: len(instrument.notes), default=None)


def chord_groups(instrument):
    """Group near-simultaneous notes into chord observations."""

    notes = sorted(instrument.notes, key=lambda note: note.start)
    groups = []
    index = 0

    while index < len(notes):
        group_start = notes[index].start
        group = []

        while (
            index < len(notes)
            and notes[index].start - group_start <= CHORD_ONSET_TOLERANCE
        ):
            group.append(notes[index])
            index += 1

        if len(group) >= MIN_CHORD_NOTES:
            groups.append({
                "start": min(note.start for note in group),
                "end": max(note.end for note in group),
                "notes": group,
                "pitch_classes": sorted({note.pitch % 12 for note in group}),
            })

    return groups


def identify_chord_track(midi, drum_track):
    """Select the strongest non-drum polyphonic track as chord reference."""

    candidates = []
    for instrument in midi.instruments:
        if instrument is drum_track or instrument.is_drum or not instrument.notes:
            continue

        chords = chord_groups(instrument)
        if not chords:
            continue

        average_polyphony = sum(
            len(chord["pitch_classes"])
            for chord in chords
        ) / len(chords)
        duration = max(note.end for note in instrument.notes) - min(
            note.start for note in instrument.notes
        )
        score = len(chords) * (1.0 + min(average_polyphony, 6.0) / 6.0)
        candidates.append((score, duration, instrument, chords))

        print(
            f"[CHORD DETECTION] {instrument.name or 'Unnamed'} | "
            f"Program: {instrument.program} | Chords: {len(chords)} | "
            f"Score: {score:.1f}"
        )

    if not candidates:
        return None, []

    _, _, instrument, chords = max(candidates, key=lambda item: (item[0], item[1]))
    return instrument, chords


def align_chord_reference(chords, grid):
    """Move each chord as a unit while retaining its duration and harmony."""

    aligned = []
    for chord in chords:
        start = nearest_grid_time(chord["start"], grid)
        duration = max(MIN_NOTE_DURATION, chord["end"] - chord["start"])
        aligned.append({
            "start": start,
            "end": start + duration,
            "pitch_classes": chord["pitch_classes"],
        })

    return sorted(aligned, key=lambda chord: chord["start"])


def synchronize_chord_track(instrument, original_chords, grid):
    """Apply the same onset shift to every note in each detected chord."""

    if not original_chords:
        return

    chord_note_ids = {
        id(note)
        for chord in original_chords
        for note in chord["notes"]
    }
    synchronized = []

    for chord in original_chords:
        target = nearest_grid_time(chord["start"], grid)
        offset = target - chord["start"]
        for note in chord["notes"]:
            synchronized.append(copy_note(
                note,
                note.start + offset,
                note.end + offset,
            ))

    for note in instrument.notes:
        if id(note) not in chord_note_ids:
            synchronized.append(copy.deepcopy(note))

    instrument.notes = sorted(synchronized, key=lambda note: note.start)


def chord_at_time(time, chords, beat_duration):
    """Find the active chord, with a bounded fallback for short gaps."""

    active = [
        chord for chord in chords
        if chord["start"] <= time <= chord["end"]
    ]
    if active:
        return min(active, key=lambda chord: abs(chord["start"] - time))

    if not chords:
        return None

    nearest = min(chords, key=lambda chord: abs(chord["start"] - time))
    if abs(nearest["start"] - time) <= beat_duration * MAX_CHORD_LOOKAHEAD_BEATS:
        return nearest
    return None


def nearest_chord_pitch(original_pitch, pitch_classes):
    """Return the closest chord tone, allowing an adjacent octave."""

    if not pitch_classes:
        return original_pitch

    candidates = []
    for octave in range(-1, 11):
        for pitch_class in pitch_classes:
            pitch = octave * 12 + pitch_class
            if 0 <= pitch <= 127:
                candidates.append(pitch)

    return min(candidates, key=lambda pitch: abs(pitch - original_pitch))


def process_melodic_track(instrument, grid, chords, beat_duration):
    """Quantize melodic attacks and force pitches into the active chord."""

    corrected = []
    moved = 0
    retuned = 0

    for note in instrument.notes:
        target_start = nearest_grid_time(note.start, grid)
        duration = max(MIN_NOTE_DURATION, note.end - note.start)
        chord = chord_at_time(target_start, chords, beat_duration)
        target_pitch = note.pitch

        if chord is not None:
            target_pitch = nearest_chord_pitch(
                note.pitch,
                chord["pitch_classes"],
            )
            if target_pitch != note.pitch:
                retuned += 1

        if abs(target_start - note.start) > 0.001:
            moved += 1

        corrected.append(copy_note(
            note,
            target_start,
            target_start + duration,
            pitch=target_pitch,
        ))

    instrument.notes = sorted(corrected, key=lambda note: note.start)
    return moved, retuned


def fix_midi(input_path):
    """Align a MIDI and return the path of the corrected file."""

    input_path = os.fspath(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    print(f"\nLoading: {input_path}")
    midi = pretty_midi.PrettyMIDI(input_path)
    duration = total_duration(midi)

    if duration <= 0:
        raise ValueError("MIDI has no valid duration.")

    drum_track = identify_drum_track(midi)
    if drum_track is None:
        raise ValueError("No drum track found.")

    beat_duration = estimate_beat_duration(midi, drum_track)
    grid = build_reference_grid(midi, drum_track, duration, beat_duration)

    print(f"[DRUM] Reference: {drum_track.name or 'Unnamed'}")
    print(f"[TIMING] Beat: {beat_duration:.3f}s")
    print(f"[TIMING] Drum onsets: {len(drum_onsets(drum_track))}")
    print(f"[TIMING] Quantization points: {len(grid)}")

    chord_track, original_chords = identify_chord_track(midi, drum_track)
    chords = []

    if chord_track is not None:
        print(
            f"[CHORD TRACK] Reference: "
            f"{chord_track.name or 'Unnamed'} | "
            f"Program: {chord_track.program}"
        )
        synchronize_chord_track(chord_track, original_chords, grid)
        chords = align_chord_reference(original_chords, grid)
        print(f"[CHORD TRACK] Chords used for autotune: {len(chords)}")
    else:
        print("[WARNING] No chord track detected; pitch correction skipped.")

    processed = 0
    total_moved = 0
    total_retuned = 0

    for instrument in midi.instruments:
        if instrument is drum_track or instrument is chord_track:
            continue
        if not instrument.notes:
            continue

        moved, retuned = process_melodic_track(
            instrument,
            grid,
            chords,
            beat_duration,
        )
        processed += 1
        total_moved += moved
        total_retuned += retuned
        print(
            f"[TRACK] {instrument.name or 'Unnamed'} | "
            f"Program: {instrument.program} | "
            f"Moved: {moved} | Retuned: {retuned}"
        )

    base, _ = os.path.splitext(input_path)
    output_path = base + "_fixed.mid"
    midi.write(output_path)

    print("\n===================================")
    print("MIDI ALIGNMENT COMPLETED")
    print("===================================")
    print(f"Output: {output_path}")
    print(f"Processed melodic tracks: {processed}")
    print(f"Moved notes: {total_moved}")
    print(f"Retuned notes: {total_retuned}")
    print("===================================\n")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align melodic MIDI tracks to drums and chord harmony."
    )
    parser.add_argument("input", type=str, help="Input MIDI file.")
    args = parser.parse_args()
    fix_midi(args.input)
