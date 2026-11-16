from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator.generate_wavefile import (  # noqa: E402
    DEFAULT_SOUNDFONT_PATH,
    midi_to_wav,
)


def _convert_file(midi_path, input_dir, output_dir):
    """Convert one MIDI file and return its generated WAV path."""

    midi_path = Path(midi_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    relative_path = midi_path.relative_to(input_dir)
    wav_path = output_dir / relative_path.with_suffix(".wav")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    midi_to_wav(
        midi_path=midi_path,
        wav_path=wav_path,
        soundfont=DEFAULT_SOUNDFONT_PATH,
    )
    return wav_path


def convert_directory(input_dir, output_dir):
    """Convert every MIDI file under input_dir while preserving its structure."""

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory not found: {input_dir}"
        )

    midi_paths = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mid"
    )

    total_files = len(midi_paths)
    worker_count = os.cpu_count() or 1
    print(f"Converting {total_files} MIDI file(s) with {worker_count} workers.")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _convert_file,
                midi_path,
                input_dir,
                output_dir,
            ): midi_path
            for midi_path in midi_paths
        }

        for file_index, future in enumerate(as_completed(futures), start=1):
            wav_path = future.result()
            print(f"Converted {file_index}/{total_files}: {wav_path}")

    return len(midi_paths)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert all .mid files under an input directory to WAV files."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing MIDI files, including nested directories.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the WAV files will be written.",
    )
    args = parser.parse_args()

    converted_count = convert_directory(args.input_dir, args.output_dir)
    print(f"Converted {converted_count} MIDI file(s).")


if __name__ == "__main__":
    main()
