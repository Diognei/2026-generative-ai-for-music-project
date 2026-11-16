
import os
import math
import pretty_midi
import numpy as np
from collections import Counter


# ============================================================
# SIX-STAGE CONFIGURATION
# ============================================================

ENABLE_DRUM_IDENTIFICATION = True
ENABLE_DRUM_CLEANING = True
ENABLE_CHORD_IDENTIFICATION = True
ENABLE_CHORD_REFINEMENT = True
ENABLE_RHYTHMIC_ALIGNMENT = True
ENABLE_HARMONIC_ALIGNMENT = True


# ============================================================
# GENERAL PARAMETERS
# ============================================================

ALIGNMENT_STRENGTH = 0.65
MAX_RHYTHMIC_SHIFT = 0.22
NOTE_GROUP_TOLERANCE = 0.08

OUTPUT_SUFFIX = "_aligned"

DRUM_PITCHES = {
    "kick": [35, 36],
    "snare": [37, 38, 39, 40],
    "tom": [41, 43, 45, 47, 48, 50],
    "hat": [42, 44, 46],
    "cymbal": [49, 51, 52, 53, 55, 57, 59]
}


# ============================================================
# DRUM GROOVE VOCABULARY
#
# Each position represents a fraction of a quarter note.
# A bar contains 4 quarter notes.
# ============================================================

DRUM_VOCABULARY = [
    {
        "name": "rock",
        "kick": [0, 2],
        "snare": [1, 3],
        "hat": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    },
    {
        "name": "rock_variation",
        "kick": [0, 1.5, 2],
        "snare": [1, 3],
        "hat": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    },
    {
        "name": "pop",
        "kick": [0, 2],
        "snare": [1, 3],
        "hat": [0, 1, 2, 3]
    },
    {
        "name": "dance",
        "kick": [0, 1, 2, 3],
        "snare": [1, 3],
        "hat": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    },
    {
        "name": "half_time",
        "kick": [0],
        "snare": [2],
        "hat": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    },
    {
        "name": "simple",
        "kick": [0, 2],
        "snare": [1, 3],
        "hat": [0, 1, 2, 3]
    },
    {
        "name": "four_on_floor",
        "kick": [0, 1, 2, 3],
        "snare": [1, 3],
        "hat": [0.5, 1.5, 2.5, 3.5]
    },
    {
        "name": "sparse",
        "kick": [0, 2],
        "snare": [1, 3],
        "hat": [0, 2]
    }
]


# ============================================================
# HARMONIC VOCABULARY
#
# Intervals in semitones from the root.
# Includes dyads, triads, and tetrads.
# ============================================================

CHORD_VOCABULARY = {
    # Dyads
    "power": [0, 7],
    "minor_dyad": [0, 3],
    "major_dyad": [0, 4],
    "sus2_dyad": [0, 2],
    "sus4_dyad": [0, 5],
    "tritone": [0, 6],

    # Triads
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],

    # Tetrads
    "dominant7": [0, 4, 7, 10],
    "major7": [0, 4, 7, 11],
    "minor7": [0, 3, 7, 10],
    "minor_major7": [0, 3, 7, 11],
    "diminished7": [0, 3, 6, 9],
    "half_diminished7": [0, 3, 6, 10],
    "augmented7": [0, 4, 8, 10],
    "augmented_major7": [0, 4, 8, 11],
    "sixth": [0, 4, 7, 9],
    "minor6": [0, 3, 7, 9],
    "dominant7sus4": [0, 5, 7, 10],
    "dominant7sus2": [0, 2, 7, 10],

    # Extensions
    "major9": [0, 4, 7, 11, 14],
    "minor9": [0, 3, 7, 10, 14],
    "dominant9": [0, 4, 7, 10, 14],
    "dominant11": [0, 4, 7, 10, 14, 17],
    "dominant13": [0, 4, 7, 10, 14, 21]
}


# ============================================================
# UTILITIES
# ============================================================

def get_beat_duration(midi):
    tempos, values = midi.get_tempo_changes()

    if len(values) == 0:
        return 0.5

    return 60.0 / float(values[0])


def get_notes(instrument):
    return sorted(
        instrument.notes,
        key=lambda n: (n.start, n.pitch)
    )


def group_notes_by_onset(notes, tolerance=NOTE_GROUP_TOLERANCE):
    groups = []

    for note in sorted(notes, key=lambda n: n.start):
        if not groups or note.start - groups[-1][0].start > tolerance:
            groups.append([note])
        else:
            groups[-1].append(note)

    return groups


def get_note_family(pitch):
    for family, pitches in DRUM_PITCHES.items():
        if pitch in pitches:
            return family

    return None


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


# ============================================================
# STAGE 1: DRUM TRACK IDENTIFICATION
# ============================================================

def score_drum_track(instrument):
    notes = instrument.notes

    if not notes:
        return -1e9

    pitches = np.array([n.pitch for n in notes])
    durations = np.array([
        max(0.001, n.end - n.start) for n in notes
    ])

    families = [
        get_note_family(n.pitch) for n in notes
    ]

    percussion_ratio = sum(
        f is not None for f in families
    ) / len(notes)

    short_ratio = np.mean(durations < 0.4)

    pitch_range = np.ptp(pitches) if len(pitches) > 1 else 0

    pitch_diversity = len(set(pitches)) / len(notes)

    family_diversity = len(
        set(f for f in families if f is not None)
    )

    # Regularity of attacks on rhythmic subdivisions.
    starts = np.array([n.start for n in notes])

    if len(starts) > 1:
        differences = np.diff(np.sort(starts))
        differences = differences[differences > 0.001]

        if len(differences):
            beat_candidates = [
                0.125, 0.25, 0.5, 1.0
            ]

            regularity = 0

            for subdivision in beat_candidates:
                errors = np.abs(
                    differences / subdivision -
                    np.round(differences / subdivision)
                )

                regularity = max(
                    regularity,
                    np.mean(errors < 0.12)
                )
        else:
            regularity = 0
    else:
        regularity = 0

    # Presence of simultaneous attacks at different pitches.
    onset_groups = group_notes_by_onset(notes)

    simultaneous_ratio = np.mean([
        len(g) >= 2 for g in onset_groups
    ])

    channel_bonus = 0.15 if instrument.is_drum else 0

    score = (
        percussion_ratio * 5.0
        + short_ratio * 1.5
        + min(pitch_range / 25.0, 1.0) * 1.0
        + min(family_diversity / 4.0, 1.0) * 1.5
        + regularity * 1.0
        + simultaneous_ratio * 0.8
        + channel_bonus
        + pitch_diversity * 0.5
    )

    return score


def identify_drum_track(midi):
    instruments = midi.instruments

    if not instruments:
        raise ValueError("O MIDI não possui pistas.")

    scores = [
        score_drum_track(inst) for inst in instruments
    ]

    # Always select a track, even if none has clear
    # drum characteristics.
    return int(np.argmax(scores))


# ============================================================
# STAGE 2: DRUM CLEANUP
# ============================================================

def drum_pattern_distance(observed, template):
    """
    Compare observed attacks with a groove.
    A lower value means greater similarity.
    """

    distance = 0

    for family in ["kick", "snare", "hat"]:
        actual = observed.get(family, [])
        expected = template[family]

        if actual and expected:
            for position in actual:
                nearest = min(
                    abs(position - target)
                    for target in expected
                )
                distance += nearest

            for position in expected:
                nearest = min(
                    [abs(position - target) for target in actual],
                    default=1.0
                )
                distance += nearest * 0.7

        elif actual:
            distance += len(actual) * 0.8

        else:
            distance += len(expected) * 0.15

    return distance


def clean_drum_track(instrument, beat):
    notes = get_notes(instrument)

    if not notes:
        return

    end_time = max(n.end for n in notes)
    bar_duration = beat * 4

    num_bars = max(
        1,
        math.ceil(end_time / bar_duration)
    )

    original_notes = notes
    new_notes = []

    for bar in range(num_bars):
        bar_start = bar * bar_duration
        bar_end = bar_start + bar_duration

        bar_notes = [
            n for n in original_notes
            if bar_start <= n.start < bar_end
        ]

        observed = {
            "kick": [],
            "snare": [],
            "hat": []
        }

        family_notes = {
            "kick": [],
            "snare": [],
            "hat": []
        }

        for note in bar_notes:
            family = get_note_family(note.pitch)

            if family in observed:
                position = (note.start - bar_start) / beat

                observed[family].append(position)
                family_notes[family].append(note)

        # Select the closest groove.
        template = min(
            DRUM_VOCABULARY,
            key=lambda t: drum_pattern_distance(observed, t)
        )

        for family in ["kick", "snare", "hat"]:
            positions = template[family]

            if family == "kick":
                default_pitch = 36
            elif family == "snare":
                default_pitch = 38
            else:
                default_pitch = 42

            available = family_notes[family].copy()

            for position in positions:
                target_time = bar_start + position * beat

                if available:
                    nearest = min(
                        available,
                        key=lambda n: abs(n.start - target_time)
                    )

                    if abs(nearest.start - target_time) <= beat * 0.25:
                        available.remove(nearest)

                        new_notes.append(
                            pretty_midi.Note(
                                velocity=nearest.velocity,
                                pitch=nearest.pitch,
                                start=max(0, target_time),
                                end=max(
                                    target_time + 0.05,
                                    min(
                                        target_time + beat * 0.35,
                                        nearest.end + target_time - nearest.start
                                    )
                                )
                            )
                        )
                        continue

                # Add an attack when necessary.
                velocity = {
                    "kick": 95,
                    "snare": 85,
                    "hat": 65
                }[family]

                new_notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=default_pitch,
                        start=target_time,
                        end=target_time + beat * 0.15
                    )
                )

        # Preserve existing cymbals and toms with light quantization.
        for note in bar_notes:
            family = get_note_family(note.pitch)

            if family in ("tom", "cymbal"):
                new_notes.append(note)

    instrument.notes = sorted(
        new_notes,
        key=lambda n: n.start
    )

    instrument.is_drum = True


# ============================================================
# STAGE 3: CHORD TRACK IDENTIFICATION
# ============================================================

def score_chord_track(instrument):
    notes = instrument.notes

    if not notes:
        return -1e9

    groups = group_notes_by_onset(notes)

    simultaneous = [
        g for g in groups if len(g) >= 2
    ]

    chord_ratio = (
        len(simultaneous) / max(1, len(groups))
    )

    average_polyphony = np.mean([
        len(g) for g in groups
    ])

    maximum_polyphony = max(
        len(g) for g in groups
    )

    # Also measure actual temporal overlap.
    overlap_count = 0

    for i, note in enumerate(notes):
        for other in notes[i + 1:]:
            if other.start >= note.end:
                break

            if other.pitch != note.pitch:
                overlap_count += 1

    score = (
        chord_ratio * 5.0
        + min(average_polyphony, 6) * 1.0
        + min(maximum_polyphony, 8) * 0.5
        + min(overlap_count / len(notes), 5) * 0.5
    )

    return score


def identify_chord_track(midi, drum_index):
    candidates = [
        i for i in range(len(midi.instruments))
        if i != drum_index
    ]

    if not candidates:
        return drum_index

    scores = [
        score_chord_track(midi.instruments[i])
        for i in candidates
    ]

    # Always select the track most likely
    # to contain chords.
    return candidates[int(np.argmax(scores))]


# ============================================================
# STAGE 4: CHORD CLEANUP AND REFINEMENT
# ============================================================

def identify_best_chord(pitches):
    pitch_classes = set(p % 12 for p in pitches)

    best_score = float("inf")
    best_chord = None

    for root in range(12):
        for name, intervals in CHORD_VOCABULARY.items():
            chord_pcs = {
                (root + interval) % 12
                for interval in intervals
            }

            missing = len(chord_pcs - pitch_classes)
            extra = len(pitch_classes - chord_pcs)

            score = (
                missing * 1.1
                + extra * 1.0
                + abs(len(chord_pcs) - len(pitch_classes)) * 0.12
            )

            if score < best_score:
                best_score = score

                best_chord = {
                    "root": root,
                    "name": name,
                    "intervals": intervals,
                    "pitch_classes": chord_pcs
                }

    return best_chord


def nearest_pitch_in_range(pitch_class, reference_pitch):
    octave = reference_pitch // 12

    candidates = [
        octave * 12 + pitch_class,
        (octave - 1) * 12 + pitch_class,
        (octave + 1) * 12 + pitch_class
    ]

    # Choose the closest note, preferring the same octave
    # in case of a tie.
    return min(
        candidates,
        key=lambda p: (
            abs(p - reference_pitch),
            abs(p // 12 - octave)
        )
    )


def refine_chord_track(instrument):
    notes = get_notes(instrument)

    if not notes:
        return

    groups = group_notes_by_onset(notes)

    refined_notes = []

    for group in groups:
        pitches = [n.pitch for n in group]

        chord = identify_best_chord(pitches)

        reference_pitch = int(round(np.mean(pitches)))

        velocity = int(round(np.mean([
            n.velocity for n in group
        ])))

        duration = float(np.median([
            n.end - n.start for n in group
        ]))

        start = min(n.start for n in group)

        chord_pitches = set()

        for pc in chord["pitch_classes"]:
            pitch = nearest_pitch_in_range(
                pc,
                reference_pitch
            )

            chord_pitches.add(
                clamp(pitch, 0, 127)
            )

        for pitch in sorted(chord_pitches):
            refined_notes.append(
                pretty_midi.Note(
                    velocity=clamp(velocity, 1, 127),
                    pitch=int(pitch),
                    start=start,
                    end=start + max(duration, 0.1)
                )
            )

    instrument.notes = sorted(
        refined_notes,
        key=lambda n: n.start
    )


# ============================================================
# STAGE 5: RHYTHMIC ALIGNMENT WITH DRUMS
# ============================================================

def get_rhythmic_reference(drum_instrument):
    """
    Return drum attacks and the midpoints
    between consecutive attacks.
    """

    drum_times = sorted(set(
        round(n.start, 5)
        for n in drum_instrument.notes
    ))

    if not drum_times:
        return []

    references = list(drum_times)

    for i in range(len(drum_times) - 1):
        midpoint = (
            drum_times[i] + drum_times[i + 1]
        ) / 2

        references.append(midpoint)

    return sorted(set(references))


def align_track_to_drums(instrument, references, beat):
    if not instrument.notes or not references:
        return

    groups = group_notes_by_onset(instrument.notes)

    for group in groups:
        original_time = float(np.mean([
            n.start for n in group
        ]))

        nearest = min(
            references,
            key=lambda t: abs(t - original_time)
        )

        distance = abs(nearest - original_time)

        # Avoid excessively large shifts.
        if distance > beat * MAX_RHYTHMIC_SHIFT:
            continue

        shift = (
            nearest - original_time
        ) * ALIGNMENT_STRENGTH

        for note in group:
            note.start = max(0, note.start + shift)
            note.end = max(
                note.start + 0.03,
                note.end + shift
            )

    instrument.notes.sort(
        key=lambda n: n.start
    )


# ============================================================
# STAGE 6: HARMONIC ALIGNMENT WITH CHORDS
# ============================================================

def get_chord_timeline(chord_instrument):
    groups = group_notes_by_onset(
        get_notes(chord_instrument)
    )

    timeline = []

    for group in groups:
        start = min(n.start for n in group)

        pitch_classes = set(
            n.pitch % 12 for n in group
        )

        if pitch_classes:
            timeline.append(
                (start, pitch_classes)
            )

    return timeline


def get_active_chord(time, timeline):
    active = None

    for start, pitch_classes in timeline:
        if start <= time:
            active = pitch_classes
        else:
            break

    return active


def align_pitch_to_chord(pitch, chord_pitch_classes):
    if pitch % 12 in chord_pitch_classes:
        return pitch

    octave = pitch // 12

    candidates = [
        octave * 12 + pc
        for pc in chord_pitch_classes
    ]

    candidates = [
        p for p in candidates
        if 0 <= p <= 127
    ]

    if not candidates:
        return pitch

    # Do not change the octave.
    # Choose exclusively among notes
    # in the original octave.
    return min(
        candidates,
        key=lambda p: abs(p - pitch)
    )


def align_track_to_harmony(instrument, chord_timeline):
    if not instrument.notes or not chord_timeline:
        return

    for note in instrument.notes:
        chord = get_active_chord(
            note.start,
            chord_timeline
        )

        if chord is None:
            continue

        note.pitch = int(
            align_pitch_to_chord(
                note.pitch,
                chord
            )
        )


# ============================================================
# MAIN FUNCTION
# ============================================================

def fix_midi(input_path):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    midi = pretty_midi.PrettyMIDI(input_path)

    if not midi.instruments:
        raise ValueError("O MIDI não possui pistas.")

    beat = get_beat_duration(midi)

    # --------------------------------------------------------
    # 1. Drum identification
    # --------------------------------------------------------

    drum_index = identify_drum_track(midi)

    print(
        f"[1] Drum track: {drum_index} "
        f"({midi.instruments[drum_index].name})"
    )

    if ENABLE_DRUM_IDENTIFICATION:
        midi.instruments[drum_index].is_drum = True

    # --------------------------------------------------------
    # 2. Drum cleanup
    # --------------------------------------------------------

    if ENABLE_DRUM_CLEANING:
        print("[2] Cleaning drum track...")

        clean_drum_track(
            midi.instruments[drum_index],
            beat
        )

    drum_instrument = midi.instruments[drum_index]

    # --------------------------------------------------------
    # 3. Chord track identification
    # --------------------------------------------------------

    chord_index = identify_chord_track(
        midi,
        drum_index
    )

    print(
        f"[3] Chord track: {chord_index} "
        f"({midi.instruments[chord_index].name})"
    )

    # --------------------------------------------------------
    # 4. Chord refinement
    # --------------------------------------------------------

    if ENABLE_CHORD_REFINEMENT:
        print("[4] Refining chords...")

        if chord_index != drum_index:
            refine_chord_track(
                midi.instruments[chord_index]
            )

    # --------------------------------------------------------
    # 5. Rhythmic alignment
    # --------------------------------------------------------

    if ENABLE_RHYTHMIC_ALIGNMENT:
        print("[5] Aligning tracks to drums...")

        references = get_rhythmic_reference(
            drum_instrument
        )

        for i, instrument in enumerate(midi.instruments):
            if i == drum_index:
                continue

            align_track_to_drums(
                instrument,
                references,
                beat
            )

    # --------------------------------------------------------
    # 6. Harmonic alignment
    # --------------------------------------------------------

    if ENABLE_HARMONIC_ALIGNMENT:
        print("[6] Aligning pitches to chords...")

        if chord_index != drum_index:
            chord_timeline = get_chord_timeline(
                midi.instruments[chord_index]
            )

            for i, instrument in enumerate(midi.instruments):
                if i in (drum_index, chord_index):
                    continue

                align_track_to_harmony(
                    instrument,
                    chord_timeline
                )

    # --------------------------------------------------------
    # SAVING
    # --------------------------------------------------------

    base, extension = os.path.splitext(input_path)

    output_path = base + OUTPUT_SUFFIX + ".mid"

    midi.write(output_path)

    print(f"\nAligned MIDI: {output_path}")

    return output_path


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fix_midi.py input.mid")
        sys.exit(1)

    output = fix_midi(sys.argv[1])

    print(f"Generated file: {output}")
