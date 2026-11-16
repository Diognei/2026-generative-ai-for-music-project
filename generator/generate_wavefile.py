from pathlib import Path
import os
import subprocess


DEFAULT_SOUNDFONT_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "PC51d.sf2"
)


def midi_to_wav(
    midi_path,
    wav_path=None,
    soundfont=DEFAULT_SOUNDFONT_PATH,
    sample_rate=44100,
):
    """
    Convert a MIDI file to WAV using FluidSynth.

    Parameters:
        midi_path:
            Path to the .mid file.

        wav_path:
            Output path for the .wav file.
            If None, use the same name as the MIDI file.

        soundfont:
            GM SoundFont used by FluidSynth.

        sample_rate:
            WAV sample rate.

    Returns:
        Path to the generated WAV file.
    """

    midi_path = os.path.abspath(midi_path)

    if wav_path is None:
        wav_path = os.path.splitext(midi_path)[0] + ".wav"

    wav_path = os.path.abspath(wav_path)

    if not os.path.isfile(midi_path):
        raise FileNotFoundError(
            f"Arquivo MIDI não encontrado: {midi_path}"
        )

    if not os.path.isfile(soundfont):
        raise FileNotFoundError(
            f"SoundFont não encontrado: {soundfont}"
        )

    command = [
        "fluidsynth",

        # Do not open a graphical interface.
        "-ni",

        # SoundFont.
        "-F", wav_path,

        # Sample rate.
        "-r", str(sample_rate),

        # SoundFont.
        soundfont,

        # MIDI.
        midi_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Erro ao converter MIDI para WAV.\n\n"
            f"Comando: {' '.join(command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    if not os.path.isfile(wav_path):
        raise RuntimeError(
            f"FluidSynth terminou sem gerar o arquivo: {wav_path}"
        )

    normalized_wav_path = wav_path + ".normalized.wav"
    normalize_command = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(sample_rate),
        normalized_wav_path,
    ]

    normalization_result = subprocess.run(
        normalize_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if normalization_result.returncode != 0:
        if os.path.isfile(normalized_wav_path):
            os.remove(normalized_wav_path)
        raise RuntimeError(
            "Erro ao normalizar o áudio WAV.\n\n"
            f"Comando: {' '.join(normalize_command)}\n\n"
            f"STDOUT:\n{normalization_result.stdout}\n\n"
            f"STDERR:\n{normalization_result.stderr}"
        )

    if not os.path.isfile(normalized_wav_path):
        raise RuntimeError(
            "A normalização terminou sem gerar o arquivo: "
            f"{normalized_wav_path}"
        )

    os.replace(normalized_wav_path, wav_path)

    return wav_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Converte um arquivo MIDI para WAV usando FluidSynth."
    )
    parser.add_argument(
        "midi_path",
        type=str,
        help="Caminho do arquivo MIDI (.mid) a ser convertido.",
    )
    parser.add_argument(
        "--wav_path",
        type=str,
        default=None,
        help="Caminho de saída do arquivo WAV (.wav). "
             "Se não fornecido, usa o mesmo nome do MIDI.",
    )
    parser.add_argument(
        "--soundfont",
        type=str,
        default=DEFAULT_SOUNDFONT_PATH,
        help="Caminho do SoundFont GM (.sf2) a ser usado. "
             f"Default: {DEFAULT_SOUNDFONT_PATH}",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Taxa de amostragem do WAV. Default: 44100 Hz",
    )

    args = parser.parse_args()

    try:
        wav_file = midi_to_wav(
            midi_path=args.midi_path,
            wav_path=args.wav_path,
            soundfont=args.soundfont,
            sample_rate=args.sample_rate,
        )
        print(f"WAV file generated successfully: {wav_file}")
    except Exception as e:
        print(f"Error: {e}")