import argparse
import os
import subprocess


def get_audio_duration(wav_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def generate_video(wav_path, png_path, output_path):
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Arquivo WAV não encontrado: {wav_path}")

    if not os.path.isfile(png_path):
        raise FileNotFoundError(f"Arquivo PNG não encontrado: {png_path}")

    duration = get_audio_duration(wav_path)

    print(f"Audio duration: {duration:.2f} seconds")

    command = [
        "ffmpeg",
        "-y",

        # Image
        "-loop", "1",
        "-framerate", "1",
        "-i", png_path,

        # Audio
        "-i", wav_path,

        # Correct odd dimensions
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",

        # 1 FPS
        "-r", "1",

        # Exact WAV duration
        "-t", str(duration),

        # Video
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",

        # Audio
        "-c:a", "aac",

        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Erro ao gerar o vídeo.")

    print(f"Video generated successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gera um vídeo MP4 a partir de uma imagem PNG e um arquivo WAV."
    )

    parser.add_argument("wav", help="Arquivo WAV")
    parser.add_argument("png", help="Imagem PNG")
    parser.add_argument("output", help="Arquivo MP4 de saída")

    args = parser.parse_args()

    generate_video(
        args.wav,
        args.png,
        args.output,
    )


if __name__ == "__main__":
    main()