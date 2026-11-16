import os
import subprocess


def wav_to_mp3(wav_path, mp3_path=None, bitrate="320k"):
    """Convert a WAV file to MP3 using the specified bitrate."""

    wav_path = os.path.abspath(wav_path)

    if mp3_path is None:
        mp3_path = os.path.splitext(wav_path)[0] + ".mp3"
    else:
        mp3_path = os.path.abspath(mp3_path)

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(
            f"Arquivo WAV não encontrado: {wav_path}"
        )

    command = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        mp3_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Erro ao converter WAV para MP3.\n\n"
            f"Comando: {' '.join(command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    if not os.path.isfile(mp3_path):
        raise RuntimeError(
            f"FFmpeg terminou sem gerar o arquivo: {mp3_path}"
        )

    return mp3_path
