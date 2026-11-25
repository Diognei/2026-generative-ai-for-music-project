"""Diversidade e similaridade simbólica entre conjuntos de MIDI (Yang & Lerch, 2020).

Protocolo de "On the evaluation of generative models in music" (Yang & Lerch,
Neural Computing and Applications, 2020):

1. Extrai de cada MIDI um conjunto de features simbólicas (alturas e ritmo).
2. Para cada feature, calcula distâncias par a par:
   - intra-set: entre músicas do MESMO conjunto. A distribuição dessas
     distâncias mede a DIVERSIDADE do conjunto (distâncias pequenas indicam
     músicas parecidas entre si, ou seja, repetição/colapso de modo);
   - inter-set: entre cada música gerada e cada música de referência.
3. Estima as densidades das distâncias com KDE gaussiano e compara:
   - Similaridade com a referência: KLD e área de sobreposição (OA) entre
     intra(referência) e inter(gerado, referência), como no artigo;
   - Diversidade relativa: razão entre as médias intra(gerado)/intra(referência)
     (~1 = tão diverso quanto as músicas humanas; <1 = menos diverso;
     >1 = mais disperso, possivelmente ruído) e OA entre as duas
     distribuições intra-set.

Uso:
    python3 evaluator/calc_diversity.py <generated_dir> <reference_dir>

As features de altura ignoram as pistas de bateria (no MIDI, a "altura" da
bateria indica o instrumento de percussão, não uma nota). As features
rítmicas usam todas as pistas. Todas as músicas são recortadas nos primeiros
--max-seconds segundos para que músicas de durações diferentes sejam
comparáveis.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pretty_midi


SUPPORTED_EXTENSIONS = {".mid", ".midi"}
DEFAULT_MAX_SECONDS = 60.0
DEFAULT_MAX_FILES = 0
KDE_GRID_POINTS = 1000
EPSILON = 1e-10

# Classes de duração de nota, em segundos. O artigo usa classes em batidas;
# aqui são segundos porque a geração não tem grade de compasso (TIME_SHIFT de
# 10 ms). A 120 BPM: 0,125 s = semicolcheia, 0,25 s = colcheia, 0,5 s =
# semínima, 1 s = mínima, 2 s = semibreve.
NOTE_LENGTH_BINS = np.array([0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, np.inf])

FEATURE_NAMES = {
    "pitch_count": "PC   pitch count",
    "pitch_class_histogram": "PCH  pitch class histogram",
    "pitch_class_transition_matrix": "PCTM pitch class transition matrix",
    "pitch_range": "PR   pitch range",
    "avg_pitch_interval": "PI   avg pitch interval",
    "note_count": "NC   note count",
    "avg_ioi": "IOI  avg inter-onset interval",
    "note_length_histogram": "NLH  note length histogram",
}


def list_midi_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"MIDI directory not found: {directory}")
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No MIDI files found in {directory}")
    return files


def sample_files(files, max_files, seed):
    if max_files is None or len(files) <= max_files:
        return files
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(files), size=max_files, replace=False))
    return [files[index] for index in indices]


def _notes(midi, max_seconds, include_drums):
    notes = [
        note
        for instrument in midi.instruments
        if include_drums or not instrument.is_drum
        for note in instrument.notes
        if max_seconds is None or note.start < max_seconds
    ]
    notes.sort(key=lambda note: (note.start, note.pitch))
    return notes


def extract_features(midi_path, max_seconds=DEFAULT_MAX_SECONDS):
    """Features de uma música. Retorna None se não houver notas melódicas."""

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    melodic = _notes(midi, max_seconds, include_drums=False)
    every_note = _notes(midi, max_seconds, include_drums=True)
    if len(melodic) < 2:
        return None

    pitches = np.array([note.pitch for note in melodic])
    pitch_classes = pitches % 12

    pitch_class_histogram = np.bincount(pitch_classes, minlength=12).astype(float)
    pitch_class_histogram /= pitch_class_histogram.sum()

    transitions = np.zeros((12, 12))
    np.add.at(transitions, (pitch_classes[:-1], pitch_classes[1:]), 1.0)
    transitions /= transitions.sum()

    onsets = np.unique([note.start for note in every_note])
    inter_onset = np.diff(onsets)

    lengths = np.array([
        (min(note.end, max_seconds) if max_seconds else note.end) - note.start
        for note in every_note
    ])
    length_histogram = np.histogram(lengths, bins=NOTE_LENGTH_BINS)[0].astype(float)
    length_histogram /= max(length_histogram.sum(), 1.0)

    return {
        "pitch_count": np.array([len(np.unique(pitches))], dtype=float),
        "pitch_class_histogram": pitch_class_histogram,
        "pitch_class_transition_matrix": transitions.ravel(),
        "pitch_range": np.array([pitches.max() - pitches.min()], dtype=float),
        "avg_pitch_interval": np.array([np.abs(np.diff(pitches)).mean()]),
        "note_count": np.array([len(every_note)], dtype=float),
        "avg_ioi": np.array([inter_onset.mean() if inter_onset.size else 0.0]),
        "note_length_histogram": length_histogram,
    }


def extract_set(files, label, max_seconds):
    """Matriz [músicas, dimensões] por feature, pulando MIDIs inválidos."""

    rows = []
    skipped = 0
    for index, path in enumerate(files, start=1):
        print(f"Loading {label} {index}/{len(files)}: {path}", flush=True)
        try:
            features = extract_features(path, max_seconds)
        except Exception as error:
            print(f"  skipped ({type(error).__name__}: {error})", flush=True)
            skipped += 1
            continue
        if features is None:
            print("  skipped (fewer than 2 melodic notes)", flush=True)
            skipped += 1
            continue
        rows.append(features)

    if len(rows) < 2:
        raise ValueError(f"Need at least 2 valid MIDI files in the {label} set.")
    stacked = {name: np.stack([row[name] for row in rows]) for name in FEATURE_NAMES}
    return stacked, len(rows), skipped


def pairwise_distances(a, b):
    """Distâncias euclidianas entre todas as linhas de a e de b."""
    squared = (
        np.sum(a**2, axis=1)[:, None]
        + np.sum(b**2, axis=1)[None, :]
        - 2.0 * a @ b.T
    )
    return np.sqrt(np.clip(squared, 0.0, None))


def intra_set_distances(features):
    distances = pairwise_distances(features, features)
    upper = np.triu_indices(len(features), k=1)  # cada par uma vez, sem i == i
    return distances[upper]


def inter_set_distances(features_a, features_b):
    return pairwise_distances(features_a, features_b).ravel()


def _bandwidth(values):
    """Regra de Silverman, com piso para distribuições quase constantes."""
    std = values.std()
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    return max(0.9 * spread * len(values) ** (-0.2), 1e-3 * (np.abs(values).max() + 1.0))


def _kde(values, grid, max_points=5000, seed=0):
    if len(values) > max_points:
        rng = np.random.default_rng(seed)
        values = rng.choice(values, size=max_points, replace=False)
    # Pelo menos 2 passos da grade: sem isso, um conjunto com distâncias todas
    # iguais (ex.: músicas idênticas) vira um pico mais estreito que a grade e
    # a densidade some (NaN).
    bandwidth = max(_bandwidth(values), 2.0 * (grid[1] - grid[0]))
    density = np.zeros_like(grid)
    for start in range(0, len(values), 1000):
        chunk = values[start:start + 1000]
        density += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bandwidth) ** 2).sum(axis=1)
    density /= len(values) * bandwidth * np.sqrt(2.0 * np.pi)
    return density, bandwidth


def compare_distributions(p_values, q_values):
    """KLD(P || Q) e área de sobreposição entre as densidades de P e Q."""

    low = min(p_values.min(), q_values.min())
    high = max(p_values.max(), q_values.max())
    margin = 3.0 * max(_bandwidth(p_values), _bandwidth(q_values))
    grid = np.linspace(low - margin, high + margin, KDE_GRID_POINTS)
    step = grid[1] - grid[0]

    p, _ = _kde(p_values, grid)
    q, _ = _kde(q_values, grid)
    p /= p.sum() * step
    q /= q.sum() * step

    kld = float(np.sum(p * np.log((p + EPSILON) / (q + EPSILON))) * step)
    overlap = float(np.sum(np.minimum(p, q)) * step)
    return max(kld, 0.0), overlap


def covariance_matrix(values):
    """Covariância com regularização numérica para poucos vetores."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] < 2:
        return np.eye(values.shape[1], dtype=float) * EPSILON

    covariance = np.cov(values, rowvar=False)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    return covariance + EPSILON * np.eye(covariance.shape[0])


def matrix_sqrt(matrix):
    """Raiz quadrada simétrica de uma matriz semidefinida positiva."""
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def frechet_midi_distance(generated_features, reference_features):
    """Fréchet Distance entre distribuições de uma feature MIDI."""
    generated_features = np.asarray(generated_features, dtype=float)
    reference_features = np.asarray(reference_features, dtype=float)

    if generated_features.ndim == 1:
        generated_features = generated_features[:, None]
    if reference_features.ndim == 1:
        reference_features = reference_features[:, None]

    if generated_features.shape[1] != reference_features.shape[1]:
        raise ValueError("Generated and reference feature dimensions differ.")

    generated_mean = generated_features.mean(axis=0)
    reference_mean = reference_features.mean(axis=0)
    generated_covariance = covariance_matrix(generated_features)
    reference_covariance = covariance_matrix(reference_features)

    generated_root = matrix_sqrt(generated_covariance)
    middle = generated_root @ reference_covariance @ generated_root
    middle_root = matrix_sqrt(middle)

    mean_distance = np.sum((generated_mean - reference_mean) ** 2)
    covariance_distance = (
        np.trace(generated_covariance)
        + np.trace(reference_covariance)
        - 2.0 * np.trace(middle_root)
    )
    return float(max(0.0, mean_distance + covariance_distance))


def evaluate(generated, reference):
    results = {}
    for name in FEATURE_NAMES:
        intra_generated = intra_set_distances(generated[name])
        intra_reference = intra_set_distances(reference[name])
        inter = inter_set_distances(generated[name], reference[name])

        kld, overlap = compare_distributions(intra_reference, inter)
        _, diversity_overlap = compare_distributions(intra_generated, intra_reference)
        reference_mean = intra_reference.mean()
        fmd = frechet_midi_distance(generated[name], reference[name])

        results[name] = {
            "intra_generated_mean": float(intra_generated.mean()),
            "intra_generated_std": float(intra_generated.std()),
            "intra_reference_mean": float(reference_mean),
            "intra_reference_std": float(intra_reference.std()),
            "inter_mean": float(inter.mean()),
            "inter_std": float(inter.std()),
            "kld_intra_ref_vs_inter": kld,
            "oa_intra_ref_vs_inter": overlap,
            "fmd": fmd,
            "diversity_ratio": float(intra_generated.mean() / max(reference_mean, EPSILON)),
            "oa_intra_gen_vs_intra_ref": diversity_overlap,
        }

    keys = ["kld_intra_ref_vs_inter", "oa_intra_ref_vs_inter", "fmd",
            "diversity_ratio", "oa_intra_gen_vs_intra_ref"]
    results["mean_over_features"] = {
        key: float(np.mean([results[name][key] for name in FEATURE_NAMES]))
        for key in keys
    }
    return results


def print_report(results, generated_count, reference_count):
    print()
    print(f"Generated songs: {generated_count} | Reference songs: {reference_count}")
    print()
    header = (
        f"{'Feature':38s} {'Intra gen':>17s} {'Intra ref':>17s} "
        f"{'Div ratio':>9s} {'OA div':>7s} {'FMD':>9s} {'KLD sim':>8s} {'OA sim':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name, label in FEATURE_NAMES.items():
        row = results[name]
        print(
            f"{label:38s} "
            f"{row['intra_generated_mean']:8.3f}+-{row['intra_generated_std']:<7.3f} "
            f"{row['intra_reference_mean']:8.3f}+-{row['intra_reference_std']:<7.3f} "
            f"{row['diversity_ratio']:9.3f} {row['oa_intra_gen_vs_intra_ref']:7.3f} "
            f"{row['fmd']:9.3f} {row['kld_intra_ref_vs_inter']:8.3f} "
            f"{row['oa_intra_ref_vs_inter']:7.3f}"
        )
    mean = results["mean_over_features"]
    print("-" * len(header))
    print(
        f"{'Mean over features':38s} {'':17s} {'':17s} "
        f"{mean['diversity_ratio']:9.3f} {mean['oa_intra_gen_vs_intra_ref']:7.3f} "
        f"{mean['fmd']:9.3f} {mean['kld_intra_ref_vs_inter']:8.3f} "
        f"{mean['oa_intra_ref_vs_inter']:7.3f}"
    )
    print()
    print("Div ratio: intra(gen)/intra(ref); ~1 = as diverse as the reference.")
    print("OA div:    overlap of intra(gen) and intra(ref); higher = closer diversity.")
    print("FMD:       Fréchet MIDI Distance; lower = closer distributions.")
    print("KLD sim:   KLD(intra(ref) || inter(gen, ref)); lower = more similar.")
    print("OA sim:    overlap of intra(ref) and inter(gen, ref); higher = more similar.")


def main():
    parser = argparse.ArgumentParser(
        description="Symbolic diversity/similarity metrics (Yang & Lerch, 2020) between MIDI directories."
    )
    parser.add_argument("input_dir", type=Path, help="Generated MIDI files.")
    parser.add_argument("reference_dir", type=Path, help="Reference MIDI files (test set).")
    parser.add_argument(
        "--max-files", type=int, default=DEFAULT_MAX_FILES,
        help=f"Random subsample per set (default: {DEFAULT_MAX_FILES}); 0 uses all files.",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
        help=f"Use only the first N seconds of each song (default: {DEFAULT_MAX_SECONDS:g}); 0 uses the whole song.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the subsample.")
    parser.add_argument("--json", type=Path, help="Optional path to save all results as JSON.")
    args = parser.parse_args()

    max_files = args.max_files or None
    max_seconds = args.max_seconds or None

    generated_files = sample_files(list_midi_files(args.input_dir), max_files, args.seed)
    reference_files = sample_files(list_midi_files(args.reference_dir), max_files, args.seed)

    generated, generated_count, generated_skipped = extract_set(
        generated_files, "generated", max_seconds)
    reference, reference_count, reference_skipped = extract_set(
        reference_files, "reference", max_seconds)

    results = evaluate(generated, reference)
    print_report(results, generated_count, reference_count)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dir": str(args.input_dir),
            "reference_dir": str(args.reference_dir),
            "max_files": max_files,
            "max_seconds": max_seconds,
            "seed": args.seed,
            "generated_count": generated_count,
            "generated_skipped": generated_skipped,
            "reference_count": reference_count,
            "reference_skipped": reference_skipped,
            "features": results,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved to: {args.json}")


if __name__ == "__main__":
    main()
