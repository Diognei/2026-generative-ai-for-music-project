import argparse
import random
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    CompositeAudioClip,
    vfx,
    afx,
)


# ============================================================
# CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_dir",
    type=Path,
    help="Directory containing the four music groups.",
)
args = parser.parse_args()

BASE_DIR = args.input_dir.expanduser().resolve()
OUTPUT_FILE = BASE_DIR / "final_video.mp4"

if not BASE_DIR.is_dir():
    raise NotADirectoryError(BASE_DIR)


GROUPS = [
    {
        "folder": "transformer-random",
        "count": 2,
        "duration": 9,
        "title": "Transformer - Random Initialized",
        "color": "#006400",
    },
    {
        "folder": "transformer-loaded",
        "count": 2,
        "duration": 12,
        "title": "Transformer - Pre-defined Structure",
        "color": "#90EE90",
    },
    {
        "folder": "lstm-random",
        "count": 1,
        "duration": 9,
        "title": "LSTM - Random Initialized",
        "color": "#00008B",
    },
    {
        "folder": "lstm-loaded",
        "count": 1,
        "duration": 9,
        "title": "LSTM - Pre-defined Structure",
        "color": "#87CEFA",
    },
]

TOTAL_DURATION = 60

# Slice mode:
# 1 = Start after initial silence.
# 2 = Random starting position.
# 3 = Use predefined starting positions from START_TIMES.
SLICE_MODE = 3
START_TIMES = [3, 0, 2, 1, 0, 0]

# Transitions
TRANSITION_DURATION = 0.5
GLOBAL_FADE_IN = 0.4
GLOBAL_FADE_OUT = 0.8

# Silence detection
SILENCE_NOISE_DB = -40
SILENCE_MIN_DURATION = 0.15
START_EPSILON = 0.05

# Audio normalization (EBU R128)
TARGET_LUFS = -16
TARGET_TRUE_PEAK = -1.5
TARGET_LRA = 11

# Title appearance
TITLE_BACKGROUND_COLOR = (0, 0, 0, 225)
TITLE_FONT_SIZE_RATIO = 0.055
TITLE_MIN_FONT_SIZE = 28
TITLE_MAX_FONT_SIZE = 64
TITLE_HORIZONTAL_PADDING = 28
TITLE_VERTICAL_PADDING = 14
TITLE_TOP_MARGIN_RATIO = 0.035
TITLE_BORDER_RADIUS = 12

# Font
TITLE_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]

# Output
FPS = 30
AUDIO_SAMPLE_RATE = 48000
ENCODING_THREADS = 20

SEED = 42


if SLICE_MODE not in (1, 2, 3):
    raise ValueError("SLICE_MODE must be 1, 2 or 3.")


# ============================================================
# FFMPEG HELPER
# ============================================================

def run_ffmpeg(command):

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg error:\n{result.stderr}"
        )

    return result.stderr


# ============================================================
# DETECT INITIAL AUDIO SILENCE
# ============================================================

def detect_audio_start(path):

    command = [
        "ffmpeg",
        "-hide_banner",
        "-i", str(path),
        "-map", "0:a:0",
        "-af",
        (
            f"silencedetect=noise={SILENCE_NOISE_DB}dB:"
            f"d={SILENCE_MIN_DURATION}"
        ),
        "-f", "null",
        "-"
    ]

    output = run_ffmpeg(command)

    silence_starts = re.findall(
        r"silence_start:\s*([0-9.]+)",
        output,
    )

    silence_ends = re.findall(
        r"silence_end:\s*([0-9.]+)",
        output,
    )

    if not silence_starts or not silence_ends:
        return 0.0

    first_silence_start = float(silence_starts[0])

    if first_silence_start > START_EPSILON:
        return 0.0

    start_time = float(silence_ends[0])

    print(f"  Initial silence removed: {start_time:.3f}s")

    return start_time


# ============================================================
# SELECT SLICE START POSITION
# ============================================================

def select_slice_start(path, video_duration, slice_duration, index):

    if SLICE_MODE == 1:

        start_time = detect_audio_start(path)

        available_duration = video_duration - start_time

        if available_duration < slice_duration:
            raise RuntimeError(
                f"{path.name} is too short after "
                f"silence removal: "
                f"{available_duration:.2f}s available, "
                f"{slice_duration:.2f}s required."
            )

        return start_time

    if SLICE_MODE == 2:

        max_start = video_duration - slice_duration

        if max_start < 0:
            raise RuntimeError(
                f"{path.name} is too short: "
                f"{video_duration:.2f}s available, "
                f"{slice_duration:.2f}s required."
            )

        start_time = random.uniform(
            0,
            max_start,
        )

        print(
            f"  Random slice start: {start_time:.3f}s"
        )

        return start_time

    if SLICE_MODE == 3:

        if index >= len(START_TIMES):
            raise RuntimeError(
                f"START_TIMES has no entry for video index {index}. "
                f"Expected at least {index + 1} entries."
            )

        start_time = float(START_TIMES[index])

        if start_time < 0:
            raise ValueError(
                f"START_TIMES[{index}] cannot be negative: {start_time}"
            )

        if start_time + slice_duration > video_duration:
            raise RuntimeError(
                f"Invalid START_TIMES[{index}] = {start_time:.3f}s for "
                f"{path.name}: the video is {video_duration:.3f}s long "
                f"and the requested slice is {slice_duration:.3f}s."
            )

        print(f"  Predefined slice start: {start_time:.3f}s")

        return start_time


# ============================================================
# NORMALIZE AUDIO
# ============================================================

def normalize_audio(
    path,
    start_time,
    duration,
    output_path,
):

    loudnorm_filter = (
        f"loudnorm="
        f"I={TARGET_LUFS}:"
        f"TP={TARGET_TRUE_PEAK}:"
        f"LRA={TARGET_LRA}"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(path),

        "-ss", str(start_time),
        "-t", str(duration),

        "-map", "0:a:0",
        "-vn",

        "-af", loudnorm_filter,

        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "2",
        "-c:a", "pcm_s16le",

        str(output_path),
    ]

    run_ffmpeg(command)


# ============================================================
# CENTER VIDEO
# ============================================================

def center_video(clip, target_width, target_height):

    x_position = (target_width - clip.w) / 2
    y_position = (target_height - clip.h) / 2

    return clip.with_position(
        (x_position, y_position)
    )


# ============================================================
# CREATE TITLE IMAGE
# ============================================================

def create_title_image(
    title,
    font_color,
    width,
    height,
    output_path,
):
    """
    Create a transparent PNG containing a centered title
    with a black background behind the text.
    """

    font_path = next(
        (
            path for path in TITLE_FONT_PATHS
            if Path(path).is_file()
        ),
        None,
    )

    if font_path is None:
        raise FileNotFoundError(
            "No suitable title font was found. "
            "Install DejaVu Sans or update TITLE_FONT_PATHS."
        )

    font_size = int(
        height * TITLE_FONT_SIZE_RATIO
    )

    font_size = max(
        TITLE_MIN_FONT_SIZE,
        min(font_size, TITLE_MAX_FONT_SIZE),
    )

    image = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(image)

    while True:

        font = ImageFont.truetype(
            font_path,
            font_size,
        )

        text_bbox = draw.textbbox(
            (0, 0),
            title,
            font=font,
        )

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        box_width = (
            text_width + 2 * TITLE_HORIZONTAL_PADDING
        )

        if box_width <= width * 0.92:
            break

        font_size -= 2

        if font_size < TITLE_MIN_FONT_SIZE:
            raise ValueError(
                f"Title does not fit in video width: {title}"
            )

    box_height = (
        text_height + 2 * TITLE_VERTICAL_PADDING
    )

    box_x = (width - box_width) // 2

    box_y = int(
        height * TITLE_TOP_MARGIN_RATIO
    )

    box_y = max(
        0,
        min(box_y, height - box_height),
    )

    draw.rounded_rectangle(
        (
            box_x,
            box_y,
            box_x + box_width,
            box_y + box_height,
        ),
        radius=TITLE_BORDER_RADIUS,
        fill=TITLE_BACKGROUND_COLOR,
    )

    text_x = box_x + (
        box_width - text_width
    ) // 2 - text_bbox[0]

    text_y = box_y + (
        box_height - text_height
    ) // 2 - text_bbox[1]

    draw.text(
        (text_x, text_y),
        title,
        font=font,
        fill=font_color,
    )

    image.save(output_path)

    return output_path


# ============================================================
# SELECT VIDEOS
# ============================================================

random.seed(SEED)

selected_videos = []

for group in GROUPS:

    folder = BASE_DIR / group["folder"]

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Group directory not found: {folder}"
        )

    files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )

    if len(files) < group["count"]:
        raise RuntimeError(
            f"Not enough videos in {folder}: "
            f"found {len(files)}, "
            f"expected {group['count']}"
        )

    selected = random.sample(
        files,
        group["count"],
    )

    for path in selected:

        selected_videos.append({
            "path": path,
            "target_duration": group["duration"],
            "title": group["title"],
            "color": group["color"],
        })


# ============================================================
# VALIDATE DURATIONS
# ============================================================

num_clips = len(selected_videos)

if SLICE_MODE == 3 and len(START_TIMES) < num_clips:
    raise ValueError(
        f"SLICE_MODE=3 requires at least {num_clips} values in START_TIMES, "
        f"but only {len(START_TIMES)} were provided."
    )

nominal_total = sum(
    item["target_duration"]
    for item in selected_videos
)

if nominal_total != TOTAL_DURATION:
    raise RuntimeError(
        f"Configured durations sum to {nominal_total}s, "
        f"expected {TOTAL_DURATION}s."
    )

if TRANSITION_DURATION * (num_clips - 1) >= TOTAL_DURATION:
    raise RuntimeError("Transition duration is too large.")

overlap_compensation = (
    TRANSITION_DURATION * (num_clips - 1) / num_clips
)


# ============================================================
# DETERMINE OUTPUT RESOLUTION
# ============================================================

print("\nAnalyzing video dimensions...")

reference_width = 0
reference_height = 0
reference_path = None

for item in selected_videos:

    clip = VideoFileClip(
        str(item["path"]),
        audio=False,
    )

    print(
        f"  {item['path'].name}: "
        f"{clip.w}x{clip.h}"
    )

    if clip.h > reference_height:
        reference_width = clip.w
        reference_height = clip.h
        reference_path = item["path"]

    clip.close()

final_width = reference_width
final_height = reference_height

print(
    f"\nReference video: {reference_path.name}"
)

print(
    f"Output resolution: {final_width}x{final_height}"
)


# ============================================================
# BUILD VIDEO, AUDIO AND TITLE CLIPS
# ============================================================

video_clips = []
audio_clips = []
title_clips = []

current_time = 0.0

with tempfile.TemporaryDirectory() as temp_dir:

    temp_dir = Path(temp_dir)

    final_video = None
    final_audio = None

    try:

        for index, item in enumerate(selected_videos):

            path = item["path"]

            duration = (
                item["target_duration"]
                + overlap_compensation
            )

            print(
                f"\n[{index + 1}/{num_clips}] "
                f"Loading: {path.name}"
            )

            print(f"  Title: {item['title']}")

            # ------------------------------------------------
            # 1. Load video
            # ------------------------------------------------

            clip = VideoFileClip(
                str(path),
                audio=False,
            )

            # ------------------------------------------------
            # 2. Select slice start position
            # ------------------------------------------------

            start_time = select_slice_start(
                path=path,
                video_duration=clip.duration,
                slice_duration=duration,
                index=index,
            )

            # ------------------------------------------------
            # 3. Trim video
            # ------------------------------------------------

            clip = clip.subclipped(
                start_time,
                start_time + duration,
            )

            # ------------------------------------------------
            # 4. Normalize audio
            # ------------------------------------------------

            temp_audio = (
                temp_dir / f"audio_{index}.wav"
            )

            print("  Normalizing audio...")

            normalize_audio(
                path=path,
                start_time=start_time,
                duration=duration,
                output_path=temp_audio,
            )

            audio = AudioFileClip(
                str(temp_audio),
                fps=AUDIO_SAMPLE_RATE,
            )

            if audio.duration < duration - 0.05:
                raise RuntimeError(
                    f"Normalized audio is too short: {path.name}"
                )

            audio = audio.subclipped(
                0,
                duration,
            )

            # ------------------------------------------------
            # 5. Video crossfade
            # ------------------------------------------------

            if index > 0:

                clip = clip.with_mask()

                clip = clip.with_effects([
                    vfx.CrossFadeIn(
                        TRANSITION_DURATION
                    )
                ])

            # ------------------------------------------------
            # 6. Center video in output canvas
            # ------------------------------------------------

            clip = center_video(
                clip,
                final_width,
                final_height,
            )

            # ------------------------------------------------
            # 7. Set video start time
            # ------------------------------------------------

            clip = clip.with_start(current_time)

            # ------------------------------------------------
            # 8. Create title overlay
            # ------------------------------------------------

            title_image_path = (
                temp_dir / f"title_{index}.png"
            )

            create_title_image(
                title=item["title"],
                font_color=item["color"],
                width=final_width,
                height=final_height,
                output_path=title_image_path,
            )

            title_clip = (
                ImageClip(str(title_image_path))
                .with_duration(duration)
                .with_start(current_time)
            )

            title_effects = []

            if index > 0:
                title_effects.append(
                    vfx.FadeIn(TRANSITION_DURATION)
                )

            if index < num_clips - 1:
                title_effects.append(
                    vfx.FadeOut(TRANSITION_DURATION)
                )

            if title_effects:
                title_clip = title_clip.with_effects(
                    title_effects
                )

            title_clips.append(title_clip)

            # ------------------------------------------------
            # 9. Audio crossfade
            # ------------------------------------------------

            audio_effects = []

            if index > 0:
                audio_effects.append(
                    afx.AudioFadeIn(
                        TRANSITION_DURATION
                    )
                )

            if index < num_clips - 1:
                audio_effects.append(
                    afx.AudioFadeOut(
                        TRANSITION_DURATION
                    )
                )

            if audio_effects:
                audio = audio.with_effects(audio_effects)

            audio = audio.with_start(current_time)

            # ------------------------------------------------
            # 10. Store clips
            # ------------------------------------------------

            video_clips.append(clip)
            audio_clips.append(audio)

            current_time += (
                duration - TRANSITION_DURATION
            )

            print(
                f"  Resolution: {clip.w}x{clip.h}"
            )

            print(
                f"  Duration: {duration:.3f}s"
            )

        # ====================================================
        # COMPOSITE VIDEO
        # ====================================================

        print("\nCompositing video...")

        final_video = CompositeVideoClip(
            video_clips + title_clips,
            size=(final_width, final_height),
            bg_color=(0, 0, 0),
        ).with_duration(TOTAL_DURATION)

        # ====================================================
        # COMPOSITE AUDIO
        # ====================================================

        print("Compositing audio...")

        final_audio = CompositeAudioClip(
            audio_clips,
        ).with_duration(TOTAL_DURATION)

        # ====================================================
        # GLOBAL FADE IN / OUT
        # ====================================================

        final_video = final_video.with_effects([
            vfx.FadeIn(GLOBAL_FADE_IN),
            vfx.FadeOut(GLOBAL_FADE_OUT),
        ])

        final_audio = final_audio.with_effects([
            afx.AudioFadeIn(GLOBAL_FADE_IN),
            afx.AudioFadeOut(GLOBAL_FADE_OUT),
        ])

        final_video = final_video.with_audio(
            final_audio
        )

        # ====================================================
        # EXPORT
        # ====================================================

        print("\nRendering final video...")

        final_video.write_videofile(
            str(OUTPUT_FILE),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            audio_fps=AUDIO_SAMPLE_RATE,
            preset="medium",
            threads=ENCODING_THREADS,
            logger="bar",
        )

        print("\nVideo successfully generated!")
        print(f"Output: {OUTPUT_FILE}")
        print(f"Resolution: {final_width}x{final_height}")
        print(f"Duration: {TOTAL_DURATION}s")
        print(f"Slice mode: {SLICE_MODE}")
        print(f"Encoding threads: {ENCODING_THREADS}")

    finally:

        if final_video is not None:
            final_video.close()

        if final_audio is not None:
            final_audio.close()

        for clip in video_clips:
            clip.close()

        for audio in audio_clips:
            audio.close()

        for clip in title_clips:
            clip.close()