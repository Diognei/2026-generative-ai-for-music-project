#!/usr/bin/env python3
"""
Adaptive MIDI Post-Processor
----------------------------
Pós-processador conservador para MIDI gerado por modelos.

Características:
- Aceita qualquer quantidade e ordem de pistas.
- Não exige que exista pista de bateria.
- Analisa programa GM, canal, pitches, durações, simultaneidade e ritmo.
- Detecta eventos com forte evidência de bateria em pistas melódicas.
- Pode criar uma pista de bateria quando necessário.
- Remove duplicatas exatas e notas inválidas.
- Opcionalmente alinha eventos não percussivos à grade observada na bateria.
- Produz um relatório JSON com o diagnóstico e as alterações.

Dependência:
    pip install pretty_midi

Uso:
    python midi_postprocessor.py entrada.mid saida.mid
    python midi_postprocessor.py entrada.mid saida.mid --report relatorio.json
    python midi_postprocessor.py entrada.mid saida.mid --align-to-drums
    python midi_postprocessor.py entrada.mid saida.mid --dry-run

O modo padrão é conservador: não força correções harmônicas ou mudanças de pitch.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pretty_midi
except ImportError as exc:
    raise SystemExit(
        "Dependência ausente. Instale com: pip install pretty_midi"
    ) from exc


# GM percussion note numbers mais comuns (General MIDI).
# São evidências, não uma regra absoluta: instrumentos melódicos também
# podem usar essas alturas.
GM_DRUM_PITCHES = {
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 80, 81
}

GM_FAMILY_NAMES = [
    "Piano", "Chromatic Percussion", "Organ", "Guitar",
    "Bass", "Strings", "Ensemble", "Brass",
    "Reed", "Pipe", "Synth Lead", "Synth Pad",
    "Synth Effects", "Ethnic", "Percussive", "Sound Effects"
]


@dataclass
class TrackProfile:
    index: int
    name: str
    program: Optional[int]
    program_name: str
    channel: Optional[int]
    is_drum_flag: bool
    note_count: int
    pitch_min: Optional[int]
    pitch_max: Optional[int]
    pitch_median: Optional[float]
    mean_duration: Optional[float]
    median_duration: Optional[float]
    polyphony_ratio: float
    mean_simultaneous_notes: float
    drum_pitch_ratio: float
    repeated_onsets_ratio: float
    role: str
    drum_score: float
    drum_confidence: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def gm_program_name(program: Optional[int]) -> str:
    if program is None:
        return "Unknown"
    try:
        return pretty_midi.program_to_instrument_name(int(program))
    except Exception:
        return f"Program {program}"


def get_track_channel(instrument: Any) -> Optional[int]:
    """pretty_midi não preserva sempre um canal único por instrumento."""
    channel = getattr(instrument, "channel", None)
    return int(channel) if channel is not None else None


def mean_simultaneous_notes(notes: Sequence[Any]) -> float:
    if not notes:
        return 0.0
    events = []
    for note in notes:
        events.append((float(note.start), 1))
        events.append((float(note.end), -1))
    # Em empate, processa NOTE_OFF antes de NOTE_ON.
    events.sort(key=lambda x: (x[0], x[1]))
    active = 0
    total = 0.0
    previous_time = events[0][0]
    for time, delta in events:
        total += active * max(0.0, time - previous_time)
        active += delta
        previous_time = time
    duration = max(0.0, max(n.end for n in notes) - min(n.start for n in notes))
    return total / duration if duration > 0 else 0.0


def calculate_polyphony_ratio(notes: Sequence[Any]) -> float:
    if len(notes) < 2:
        return 0.0
    events = []
    for n in notes:
        events.extend([(n.start, 1), (n.end, -1)])
    events.sort(key=lambda x: (x[0], x[1]))
    active = 0
    overlapping_time = 0.0
    total_time = 0.0
    prev = events[0][0]
    for t, delta in events:
        dt = max(0.0, t - prev)
        total_time += dt
        if active >= 2:
            overlapping_time += dt
        active += delta
        prev = t
    return overlapping_time / total_time if total_time > 0 else 0.0


def onset_repetition_ratio(notes: Sequence[Any], tolerance: float = 0.035) -> float:
    """Proporção de onsets próximos de uma repetição temporal."""
    if len(notes) < 4:
        return 0.0
    onsets = sorted(float(n.start) for n in notes)
    diffs = [b - a for a, b in zip(onsets, onsets[1:]) if b - a > 0.005]
    if len(diffs) < 3:
        return 0.0
    rounded = [round(d / tolerance) for d in diffs]
    counts = Counter(rounded)
    repeated = sum(count for count in counts.values() if count >= 2)
    return repeated / len(diffs)


def infer_role(
    program: Optional[int],
    pitch_median: Optional[float],
    polyphony: float,
    mean_simul: float,
    mean_duration: Optional[float],
    drum_score: float,
) -> str:
    if drum_score >= 0.72:
        return "drums"
    family = program // 8 if program is not None else None
    if family == 4 or (pitch_median is not None and pitch_median < 48 and polyphony < 0.25):
        return "bass"
    if polyphony >= 0.22 or mean_simul >= 2.0:
        return "harmony"
    if mean_duration is not None and mean_duration >= 1.2:
        return "sustained"
    if family in (5, 6, 7, 11):
        return "sustained"
    if len([x for x in [program] if x is not None]) and family == 14:
        return "percussive"
    return "melody"


def analyze_track(index: int, instrument: Any) -> TrackProfile:
    notes = list(instrument.notes)
    pitches = [int(n.pitch) for n in notes]
    durations = [max(0.0, float(n.end - n.start)) for n in notes]
    program = None if instrument.is_drum else int(instrument.program)
    channel = get_track_channel(instrument)
    drum_flag = bool(instrument.is_drum)

    drum_pitch_ratio = (
        sum(1 for p in pitches if p in GM_DRUM_PITCHES) / len(pitches)
        if pitches else 0.0
    )
    short_ratio = (
        sum(1 for d in durations if d <= 0.22) / len(durations)
        if durations else 0.0
    )
    repetition = onset_repetition_ratio(notes)
    polyphony = calculate_polyphony_ratio(notes)
    mean_simul = mean_simultaneous_notes(notes)
    median_pitch = statistics.median(pitches) if pitches else None
    mean_duration = statistics.mean(durations) if durations else None

    # Evidências ponderadas. O canal de bateria/is_drum é uma evidência forte.
    score = 0.0
    if drum_flag:
        score = 1.0
    else:
        if channel == 9:
            score += 0.72
        score += 0.15 * drum_pitch_ratio
        score += 0.08 * short_ratio
        score += 0.10 * repetition
        # Pistas muito polifônicas são menos provavelmente kits de bateria.
        score -= 0.12 * polyphony
        score = clamp(score, 0.0, 1.0)

    if drum_flag or channel == 9 or score >= 0.72:
        confidence = "high"
    elif score >= 0.40:
        confidence = "medium"
    else:
        confidence = "low"

    role = infer_role(
        program, median_pitch, polyphony, mean_simul, mean_duration, score
    )

    return TrackProfile(
        index=index,
        name=str(instrument.name or f"Track {index}"),
        program=program,
        program_name="Drum Kit" if drum_flag else gm_program_name(program),
        channel=channel,
        is_drum_flag=drum_flag,
        note_count=len(notes),
        pitch_min=min(pitches) if pitches else None,
        pitch_max=max(pitches) if pitches else None,
        pitch_median=median_pitch,
        mean_duration=mean_duration,
        median_duration=statistics.median(durations) if durations else None,
        polyphony_ratio=polyphony,
        mean_simultaneous_notes=mean_simul,
        drum_pitch_ratio=drum_pitch_ratio,
        repeated_onsets_ratio=repetition,
        role=role,
        drum_score=score,
        drum_confidence=confidence,
    )


def remove_invalid_and_duplicate_notes(
    instrument: Any,
    report: Dict[str, Any],
    duplicate_tolerance: float = 0.002,
) -> None:
    """Remove notas inválidas e duplicatas exatas na mesma pista."""
    cleaned = []
    seen = set()
    removed_invalid = 0
    removed_duplicates = 0

    for note in instrument.notes:
        start = float(note.start)
        end = float(note.end)
        pitch = int(note.pitch)
        velocity = int(note.velocity)

        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or pitch < 0
            or pitch > 127
            or velocity < 1
            or velocity > 127
        ):
            removed_invalid += 1
            continue

        key = (
            pitch,
            round(start / duplicate_tolerance),
            round(end / duplicate_tolerance),
        )
        if key in seen:
            removed_duplicates += 1
            continue
        seen.add(key)
        cleaned.append(note)

    instrument.notes = sorted(cleaned, key=lambda n: (n.start, n.pitch))
    report["invalid_notes_removed"] += removed_invalid
    report["duplicate_notes_removed"] += removed_duplicates


def find_drum_destination(midi: Any) -> Optional[Any]:
    candidates = [
        inst for inst in midi.instruments
        if inst.is_drum or get_track_channel(inst) == 9
    ]
    if candidates:
        return max(candidates, key=lambda x: len(x.notes))
    return None


def move_confident_drum_events(
    midi: Any,
    profiles: List[TrackProfile],
    report: Dict[str, Any],
    threshold: float = 0.82,
) -> None:
    """
    Move apenas notas com evidência alta de bateria.
    Não tenta adivinhar notas melódicas em uma pista já marcada como drums,
    pois os pitches GM de bateria são ambíguos sem contexto adicional.
    """
    drum_tracks = [
        midi.instruments[p.index]
        for p in profiles
        if p.is_drum_flag or p.channel == 9
    ]
    destination = find_drum_destination(midi)

    transfers: List[Tuple[Any, Any]] = []
    for profile in profiles:
        source = midi.instruments[profile.index]
        if source is destination or source.is_drum or profile.channel == 9:
            continue
        if profile.drum_score < threshold:
            continue

        # Uma pista inteira só é movida se a classificação for muito forte.
        # Caso contrário, só transferimos notas em pitches GM quando a pista
        # tem padrão percussivo claro e baixa evidência de função harmônica.
        eligible_track = (
            profile.drum_score >= threshold
            and profile.drum_pitch_ratio >= 0.45
            and profile.polyphony_ratio < 0.35
        )
        if not eligible_track:
            continue

        for note in list(source.notes):
            if note.pitch in GM_DRUM_PITCHES:
                transfers.append((source, note))

    if not transfers:
        return

    if destination is None:
        destination = pretty_midi.Instrument(
            program=0, is_drum=True, name="Recovered Drums"
        )
        midi.instruments.append(destination)
        report["drum_track_created"] = True

    for source, note in transfers:
        destination.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=float(note.start),
                end=float(note.end),
            )
        )
        source.notes.remove(note)
        report["drum_notes_moved"] += 1

    destination.notes.sort(key=lambda n: (n.start, n.pitch))


def estimate_grid_period(drum_instruments: Sequence[Any]) -> Optional[float]:
    onsets = sorted(
        float(n.start)
        for inst in drum_instruments
        for n in inst.notes
    )
    if len(onsets) < 4:
        return None

    diffs = [b - a for a, b in zip(onsets, onsets[1:]) if 0.04 <= b - a <= 1.0]
    if len(diffs) < 3:
        return None

    # Usa diferenças pequenas e robustas para aproximar a subdivisão dominante.
    candidates = [d for d in diffs if d <= statistics.median(diffs)]
    if not candidates:
        return None
    period = statistics.median(candidates)
    return period if 0.06 <= period <= 0.75 else None


def align_to_drum_grid(
    midi: Any,
    report: Dict[str, Any],
    max_shift: float = 0.045,
    strength: float = 0.65,
) -> None:
    """
    Ajuste opcional e suave de onsets à grade estimada da bateria.
    Preserva a bateria e não desloca notas mais que max_shift segundos.
    """
    drums = [i for i in midi.instruments if i.is_drum or get_track_channel(i) == 9]
    if not drums:
        report["rhythm_alignment_skipped"] = "No drum track detected"
        return

    period = estimate_grid_period(drums)
    if period is None:
        report["rhythm_alignment_skipped"] = "Could not estimate drum grid"
        return

    anchor = min(n.start for inst in drums for n in inst.notes)
    changed = 0
    for inst in midi.instruments:
        if inst in drums:
            continue
        for note in inst.notes:
            relative = note.start - anchor
            nearest = anchor + round(relative / period) * period
            delta = nearest - note.start
            if abs(delta) <= max_shift:
                shift = delta * clamp(strength, 0.0, 1.0)
                note.start = max(0.0, note.start + shift)
                note.end = max(note.start + 0.01, note.end + shift)
                changed += 1

    report["rhythm_notes_adjusted"] = changed
    report["estimated_drum_grid_seconds"] = period


def build_report(midi: Any) -> Tuple[List[TrackProfile], Dict[str, Any]]:
    profiles = [
        analyze_track(i, inst)
        for i, inst in enumerate(midi.instruments)
    ]
    report: Dict[str, Any] = {
        "input_track_count": len(midi.instruments),
        "tracks": [asdict(p) for p in profiles],
        "invalid_notes_removed": 0,
        "duplicate_notes_removed": 0,
        "drum_notes_moved": 0,
        "drum_track_created": False,
        "rhythm_notes_adjusted": 0,
    }
    return profiles, report


def process_midi(
    input_path: Path,
    output_path: Path,
    report_path: Optional[Path] = None,
    align_drums: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    midi = pretty_midi.PrettyMIDI(str(input_path))
    profiles, report = build_report(midi)

    for instrument in midi.instruments:
        remove_invalid_and_duplicate_notes(instrument, report)

    # Recalcula os perfis depois da limpeza.
    profiles = [
        analyze_track(i, inst)
        for i, inst in enumerate(midi.instruments)
    ]
    report["tracks_after_cleanup"] = [asdict(p) for p in profiles]

    move_confident_drum_events(midi, profiles, report)

    if align_drums:
        align_to_drum_grid(midi, report)

    report["output_track_count"] = len(midi.instruments)
    report["output_note_count"] = sum(len(i.notes) for i in midi.instruments)

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(output_path))
        report["output_file"] = str(output_path)
    else:
        report["dry_run"] = True

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return report


def print_summary(report: Dict[str, Any]) -> None:
    print("\n=== MIDI Post-Processor ===")
    print(f"Pistas de entrada: {report['input_track_count']}")
    print(f"Pistas de saída:   {report['output_track_count']}")
    print(f"Notas de saída:    {report['output_note_count']}")
    print(f"Duplicatas removidas: {report['duplicate_notes_removed']}")
    print(f"Notas inválidas removidas: {report['invalid_notes_removed']}")
    print(f"Notas movidas para bateria: {report['drum_notes_moved']}")
    print(f"Pista de bateria criada: {report['drum_track_created']}")
    print(f"Notas alinhadas: {report['rhythm_notes_adjusted']}")

    print("\nDiagnóstico das pistas:")
    for track in report["tracks"]:
        print(
            f"- [{track['index']}] {track['name']}: "
            f"{track['program_name']} | role={track['role']} | "
            f"notes={track['note_count']} | "
            f"drum_score={track['drum_score']:.2f} "
            f"({track['drum_confidence']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pós-processador adaptativo e conservador de arquivos MIDI."
    )
    parser.add_argument("input", type=Path, help="MIDI de entrada")
    parser.add_argument("output", type=Path, help="MIDI de saída")
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Caminho opcional para salvar diagnóstico JSON",
    )
    parser.add_argument(
        "--align-to-drums", action="store_true",
        help="Alinha suavemente onsets às subdivisões estimadas da bateria",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analisa e calcula correções sem salvar o MIDI",
    )
    return parser.parse_args()


def fix_midi(input_path: Path) -> str:
    """Processa um MIDI e retorna o caminho da versão corrigida."""

    input_path = Path(input_path)
    output_path = input_path.with_name(f"{input_path.stem}_fixed.mid")
    process_midi(input_path, output_path)
    return str(output_path)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Arquivo não encontrado: {args.input}")

    report = process_midi(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        align_drums=args.align_to_drums,
        dry_run=args.dry_run,
    )
    print_summary(report)


if __name__ == "__main__":
    main()
