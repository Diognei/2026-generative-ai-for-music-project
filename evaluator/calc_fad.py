import argparse
from pathlib import Path

import numpy as np

from audio_features import (
    extract_embeddings_with_progress,
    list_audio_files,
)


def covariance(embeddings):
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[1], embeddings.shape[1]), dtype=np.float64)
    return np.cov(embeddings, rowvar=False)


def matrix_sqrt(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def frechet_distance(input_embeddings, reference_embeddings):
    input_mean = input_embeddings.mean(axis=0)
    reference_mean = reference_embeddings.mean(axis=0)
    input_covariance = covariance(input_embeddings)
    reference_covariance = covariance(reference_embeddings)

    covariance_root = matrix_sqrt(input_covariance)
    middle = covariance_root @ reference_covariance @ covariance_root
    covariance_product_root = matrix_sqrt(middle)
    mean_distance = np.sum((input_mean - reference_mean) ** 2)
    covariance_distance = (
        np.trace(input_covariance)
        + np.trace(reference_covariance)
        - 2.0 * np.trace(covariance_product_root)
    )
    return float(max(0.0, mean_distance + covariance_distance))


def calculate_fad(input_dir, reference_dir):
    input_files = list_audio_files(input_dir)
    reference_files = list_audio_files(reference_dir)
    print(f"Found {len(input_files)} input WAV file(s).", flush=True)
    print(f"Found {len(reference_files)} reference WAV file(s).", flush=True)

    reference_embeddings = np.concatenate(
        extract_embeddings_with_progress(
            reference_files,
            "Loading reference",
        ),
        axis=0,
    )
    input_embeddings = extract_embeddings_with_progress(
        input_files,
        "Loading input",
    )

    scores = []
    for file_index, (path, embeddings) in enumerate(
        zip(input_files, input_embeddings),
        start=1,
    ):
        print(
            f"Calculating FAD {file_index}/{len(input_files)}: {path}",
            flush=True,
        )
        scores.append(frechet_distance(embeddings, reference_embeddings))
    return np.asarray(scores, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate spectral Fréchet Audio Distance between WAV directories."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    args = parser.parse_args()

    scores = calculate_fad(args.input_dir, args.reference_dir)
    print(f"FAD: {scores.mean():.6f} +- {scores.std():.6f}")


if __name__ == "__main__":
    main()
