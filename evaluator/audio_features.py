from pathlib import Path
import wave

import numpy as np


SUPPORTED_EXTENSIONS = {".wav"}
TARGET_SAMPLE_RATE = 16000
FRAME_SIZE = 1024
HOP_SIZE = 512
BAND_COUNT = 64


def list_audio_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Audio directory not found: {directory}")
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No WAV files found in {directory}")
    return files


def load_wav(path):
    path = Path(path)
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_audio = wav_file.readframes(frame_count)

    if sample_width == 1:
        audio = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32)
        audio /= 32768.0
    elif sample_width == 3:
        bytes_array = np.frombuffer(raw_audio, dtype=np.uint8).reshape(-1, 3)
        signed = (
            bytes_array[:, 0].astype(np.int32)
            | (bytes_array[:, 1].astype(np.int32) << 8)
            | (bytes_array[:, 2].astype(np.int32) << 16)
        )
        signed[signed >= 2**23] -= 2**24
        audio = signed.astype(np.float32) / 2**23
    elif sample_width == 4:
        audio = np.frombuffer(raw_audio, dtype="<i4").astype(np.float32)
        audio /= 2**31
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if audio.size == 0:
        raise ValueError(f"WAV file is empty: {path}")

    return audio, sample_rate


def resample_audio(audio, source_rate, target_rate=TARGET_SAMPLE_RATE):
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    target_size = max(1, round(audio.size * target_rate / source_rate))
    source_positions = np.linspace(0, audio.size - 1, num=audio.size)
    target_positions = np.linspace(0, audio.size - 1, num=target_size)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _band_edges(bin_count, sample_rate):
    frequencies = np.fft.rfftfreq(FRAME_SIZE, d=1.0 / sample_rate)
    lower = np.linspace(50.0, sample_rate / 2.0, BAND_COUNT + 1)
    edges = np.searchsorted(frequencies, lower)
    edges[0] = max(1, edges[0])
    edges[-1] = min(bin_count, edges[-1])
    return edges


def extract_embeddings(path):
    audio, sample_rate = load_wav(path)
    audio = resample_audio(audio, sample_rate)
    if audio.size < FRAME_SIZE:
        audio = np.pad(audio, (0, FRAME_SIZE - audio.size))

    frame_count = 1 + (audio.size - FRAME_SIZE) // HOP_SIZE
    frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME_SIZE)[::HOP_SIZE]
    frames = frames[:frame_count]
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    edges = _band_edges(spectrum.shape[1], TARGET_SAMPLE_RATE)

    bands = np.empty((spectrum.shape[0], BAND_COUNT), dtype=np.float64)
    for index in range(BAND_COUNT):
        start = edges[index]
        end = max(start + 1, edges[index + 1])
        bands[:, index] = spectrum[:, start:end].mean(axis=1)

    bands = np.log1p(bands)
    bands /= np.maximum(bands.sum(axis=1, keepdims=True), 1e-12)
    return bands


def aggregate_embeddings(paths):
    embeddings = [extract_embeddings(path) for path in paths]
    return np.concatenate(embeddings, axis=0)


def distribution(embeddings):
    values = np.asarray(embeddings, dtype=np.float64).mean(axis=0)
    values = np.clip(values, 1e-12, None)
    return values / values.sum()
