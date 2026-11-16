
from pathlib import Path
import argparse
import random
import shutil
import torch
import pretty_midi

from vocabulary import Vocabulary
from model import MusicTransformer
from constraints import ConstrainedDecoder
from midi_events import midi_to_tokens, tokens_to_midi, TIME_STEP
from generate_pianoroll import generate_pianoroll
from generate_wavefile import midi_to_wav
from generate_mp3file import wav_to_mp3
from generate_videofile import generate_video
from align_midi import fix_midi

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/model_best.pt")
USE_REF_FILE = False
ALIGN_FINAL_MIDI = True


#BEST CONFIGURATION:
TEMPERATURE = 1.1
TOP_K = 32
TOP_P = 0.9
REPETITION_PENALTY = 1.08
REPETITION_WINDOW = 96
MAX_NEW_TOKENS = 4000
MAX_TRACKS = 6
DATA_DIR = Path("data")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOUNDFONT_PATH = PROJECT_ROOT / "assets" / "PC51d.sf2"

#Override default parameters:
#TEMPERATURE = 0.75
#MAX_NEW_TOKENS = 4096
#TOP_K = 32
#TOP_P = 0.9
#REPETITION_PENALTY = 1.0
#REPETITION_WINDOW = 128

#Override default parameters:
#MAX_NEW_TOKENS = 12000
#TOP_K = 64
#TOP_P = 1
#REPETITION_PENALTY = 1.0
#REPETITION_WINDOW = 128

# Maximum note duration: 1 bar at 120 BPM (4/4), with a 10 ms TIME_STEP.
DEFAULT_TEMPO_BPM = 120.0
BEATS_PER_BAR = 4
MAX_NOTE_DURATION_STEPS = round(
    (60.0 / DEFAULT_TEMPO_BPM) * BEATS_PER_BAR / TIME_STEP
)


def create_output_directory():
    """Create and return the next available output directory."""

    output_root = PROJECT_ROOT / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        output_dir = output_root / f"out{index}"
        if not output_dir.exists():
            output_dir.mkdir()
            return output_dir
        index += 1


def sample_token(
    logits,
    allowed_tensor,
    temperature,
    recent_tokens,
    top_k=TOP_K,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY
):
    """Sample with temperature, top-k, nucleus, and repetition penalty."""

    scaled_logits = logits[allowed_tensor].clone()
    allowed_ids = set(allowed_tensor.tolist())

    for index, token_id in enumerate(allowed_tensor.tolist()):
        if token_id in recent_tokens and token_id in allowed_ids:
            if scaled_logits[index] > 0:
                scaled_logits[index] /= repetition_penalty
            else:
                scaled_logits[index] *= repetition_penalty

    scaled_logits = scaled_logits / temperature

    if top_k > 0 and top_k < scaled_logits.numel():
        threshold = torch.topk(scaled_logits, top_k).values[-1]
        scaled_logits[scaled_logits < threshold] = -torch.inf

    probabilities = torch.softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities,
            descending=True
        )
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_probabilities[remove] = 0.0
        probabilities = torch.zeros_like(probabilities)
        probabilities.scatter_(0, sorted_indices, sorted_probabilities)
        probabilities /= probabilities.sum()

    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Probabilidades invalidas durante a amostragem.")

    selected = torch.multinomial(probabilities, num_samples=1)
    return allowed_tensor[selected].item()


def initial_sequence(vocab, data_dir=DATA_DIR):
    """Initialize generation with the dataset track structure."""

    if not USE_REF_FILE:
        return _initial_sequence_no_reference(vocab)

    midi_paths = sorted(data_dir.rglob("*.mid"))
    midi_paths += sorted(data_dir.rglob("*.midi"))

    if not midi_paths:
        raise FileNotFoundError(
            "Nenhum MIDI encontrado em ./data/ para inicializar as pistas."
        )

    reference_midi_path = random.choice(midi_paths)
    midi = pretty_midi.PrettyMIDI(str(reference_midi_path))
    instruments = [
        instrument
        for instrument in midi.instruments
        if instrument.notes
    ]
    instruments = instruments[:MAX_TRACKS]

    if not instruments:
        raise ValueError(
            f"Nenhuma pista com notas encontrada em {reference_midi_path}"
        )

    track_plan = [
        (instrument.is_drum, instrument.program)
        for instrument in instruments
    ]
    max_track_duration = max(
        note.end
        for instrument in instruments
        for note in instrument.notes
    )
    reference_tokens = midi_to_tokens(reference_midi_path, vocab)
    reference_budgets = []
    current_budget = 0
    in_track = False

    for token_id in reference_tokens:
        if token_id == vocab.track_start_id:
            current_budget = 1
            in_track = True
        elif in_track:
            current_budget += 1
            if token_id == vocab.track_end_id:
                reference_budgets.append(max(32, current_budget + 32))
                in_track = False

    total_budget = max(1, MAX_NEW_TOKENS - len(track_plan) * 8)
    reference_total = sum(reference_budgets)
    track_budgets = [
        max(
            8,
            round(total_budget * budget / reference_total)
        )
        for budget in reference_budgets
    ]

    while sum(track_budgets) > total_budget:
        largest = max(range(len(track_budgets)), key=track_budgets.__getitem__)
        if track_budgets[largest] <= 8:
            break
        track_budgets[largest] -= 1

    sequence = [vocab.music_start_id]
    for instrument in instruments:
        sequence.append(vocab.track_start_id)
        sequence.append(
            vocab.is_drum_id if instrument.is_drum else vocab.is_melodic_id
        )
        if not instrument.is_drum:
            sequence.append(vocab.encode(f"PROGRAM_{instrument.program}"))
        sequence.append(vocab.track_end_id)

    return sequence, track_plan, max_track_duration, track_budgets


# ------------------------------------------------------------
# Instrument reference for generation without a MIDI file.
# ------------------------------------------------------------
# Each category: (is_drum, [possible GM programs])
_REF_CATEGORIES = {
    "drums": (True, [0]),
    "ensemble": (False, list(range(40, 48)) + list(range(48, 56)) + list(range(72, 80))),
    "pad": (False, list(range(88, 96))),
    "bass": (False, list(range(32, 40))),
    "organ": (False, list(range(16, 24))),
    "piano": (False, list(range(0, 8)) + list(range(8, 16))),
    "guitar": (False, list(range(24, 32)) + list(range(104, 112))),
    "brass": (False, list(range(56, 64)) + list(range(64, 72))),
    "lead": (False, list(range(80, 88))),
    "percussive": (False, list(range(112, 120))),
}

_REF_NON_DRUM_ORDER = [
    "ensemble", "pad", "bass", "organ", "piano",
    "guitar", "brass", "lead", "percussive",
]


def _initial_sequence_no_reference(vocab):
    """Create the track plan and initial sequence without a reference file.

    The first track is always drums (channel 10). The remaining tracks use
    up to 10 reference categories (ensemble, pad, bass, organ, piano,
    guitar, brass, lead, percussive), without repetition when the total is
    <= 11. When > 11, categories may repeat.

    Return only [MUSIC_START] as the initial sequence; _generate_no_reference
    opens each track in track_plan one at a time, ensuring that all tracks
    are filled with notes generated by the model.
    """

    max_instruments = max(1, MAX_TRACKS)
    min_instruments = min(4, max_instruments)
    n_instruments = random.randint(min_instruments, max_instruments)

    # Track 1 is always drums (channel 10).
    track_plan = [(True, 0)]

    max_unique_instruments = len(_REF_NON_DRUM_ORDER) + 1
    if n_instruments <= max_unique_instruments:
        chosen = random.sample(_REF_NON_DRUM_ORDER, n_instruments - 1)
    else:
        chosen = list(_REF_NON_DRUM_ORDER)
        remaining = n_instruments - 1 - len(chosen)
        chosen.extend(random.choices(_REF_NON_DRUM_ORDER, k=remaining))

    for category in chosen:
        is_drum, programs = _REF_CATEGORIES[category]
        program = random.choice(programs)
        track_plan.append((is_drum, program))

    return [vocab.music_start_id], track_plan, None, None


@torch.no_grad()
def generate(
    model,
    vocab,
    device,
    temperature=1.0,
    max_new_tokens=4096
):
    if temperature <= 0:
        raise ValueError("Temperature deve ser maior que zero.")

    model.eval()

    if not USE_REF_FILE:
        (
            _,
            track_plan,
            _,
            _,
        ) = initial_sequence(vocab)
        return _generate_no_reference(
            model,
            vocab,
            device,
            track_plan,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    (
        sequence,
        track_plan,
        max_track_duration,
        track_budgets
    ) = initial_sequence(vocab)

    constraints = ConstrainedDecoder(
        vocab,
        max_tracks=MAX_TRACKS,
        max_polyphony=32,
        track_plan=track_plan,
        max_track_duration=max_track_duration,
        track_budgets=track_budgets,
        max_note_duration_steps=MAX_NOTE_DURATION_STEPS,
    )

    for step in range(max_new_tokens):

        context = sequence[-model.block_size:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device
        )

        logits = model(x)[0, -1, :]

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"Logits inválidos na etapa {step}."
            )

        allowed = constraints.allowed_next(sequence)

        if not allowed:
            raise RuntimeError(
                f"Nenhum token permitido na etapa {step}."
            )

        allowed_tensor = torch.tensor(
            allowed,
            dtype=torch.long,
            device=device
        )

        if temperature == 0:
            allowed_logits = logits[allowed_tensor]
            next_token = allowed_tensor[
                torch.argmax(allowed_logits)
            ].item()

        else:
            next_token = sample_token(
                logits,
                allowed_tensor,
                temperature,
                sequence[-REPETITION_WINDOW:]
            )

        sequence.append(next_token)

        if next_token == vocab.music_end_id:
            break

    else:
        sequence = constraints.force_close(sequence)

    return sequence


def _generate_no_reference(
    model,
    vocab,
    device,
    track_plan,
    temperature=1.0,
    max_new_tokens=4096,
):
    """Generate a song by opening each track in track_plan one at a time.

    Unlike the standard generate() function, which lets the model choose when
    to switch tracks, this function ensures that ALL planned tracks are opened
    and filled with notes. This prevents the song from containing only drums.
    """

    constraints = ConstrainedDecoder(
        vocab,
        max_tracks=MAX_TRACKS,
        max_polyphony=32,
        track_plan=track_plan,
        max_track_duration=None,
        track_budgets=None,
        max_note_duration_steps=MAX_NOTE_DURATION_STEPS,
    )

    sequence = [vocab.music_start_id]

    # The global format requires all headers before the events.
    for is_drum, program in track_plan:
        sequence.append(vocab.track_start_id)
        if is_drum:
            sequence.append(vocab.is_drum_id)
        else:
            sequence.extend([
                vocab.is_melodic_id,
                vocab.encode(f"PROGRAM_{program}"),
            ])
        sequence.append(vocab.track_end_id)

    tokens_per_track = max(64, max_new_tokens // len(track_plan))
    total_generated = 0

    for track_idx, (is_drum, program) in enumerate(track_plan):
        sequence.append(vocab.encode(f"TRACK_SELECT_{track_idx}"))

        # Token limit for this track
        if track_idx < len(track_plan) - 1:
            track_limit = tokens_per_track
        else:
            track_limit = max_new_tokens - total_generated

        track_tokens = 0

        while track_tokens < track_limit and total_generated < max_new_tokens:
            context = sequence[-model.block_size:]
            x = torch.tensor(
                [context],
                dtype=torch.long,
                device=device,
            )

            logits = model(x)[0, -1, :]

            if not torch.isfinite(logits).all():
                break

            allowed = [
                token_id
                for token_id in constraints.allowed_next(sequence)
                if token_id != vocab.music_end_id
            ]
            if not allowed:
                break

            allowed_tensor = torch.tensor(
                allowed,
                dtype=torch.long,
                device=device,
            )

            if temperature == 0:
                allowed_logits = logits[allowed_tensor]
                next_token = allowed_tensor[
                    torch.argmax(allowed_logits)
                ].item()
            else:
                next_token = sample_token(
                    logits,
                    allowed_tensor,
                    temperature,
                    sequence[-REPETITION_WINDOW:],
                )

            sequence.append(next_token)
            total_generated += 1
            track_tokens += 1

            if next_token == vocab.track_end_id:
                break

    # Close the song
    sequence.append(vocab.music_end_id)

    return sequence


def generate_media_files(midi_path):
    """Generate media for the original MIDI and its aligned version."""

    midi_path = Path(midi_path)
    midi_paths = [midi_path]
    if ALIGN_FINAL_MIDI:
        try:
            midi_paths.append(Path(fix_midi(midi_path)))
        except ValueError as error:
            print(f"Alignment skipped: {error}")

    for current_midi_path in midi_paths:
        output_dir = current_midi_path.parent
        pianoroll_path = output_dir / f"{current_midi_path.stem}_pianoroll.png"
        wav_path = output_dir / f"{current_midi_path.stem}.wav"
        mp3_path = output_dir / f"{current_midi_path.stem}.mp3"
        video_path = output_dir / f"{current_midi_path.stem}.mp4"

        generate_pianoroll(current_midi_path, pianoroll_path)
        midi_to_wav(current_midi_path, wav_path, soundfont=SOUNDFONT_PATH)
        wav_to_mp3(wav_path, mp3_path, bitrate="320k")
        generate_video(wav_path, pianoroll_path, video_path)

        print(f"Pianoroll generated: {pianoroll_path}")
        print(f"WAV generated: {wav_path}")
        print(f"MP3 generated: {mp3_path}")
        print(f"Video generated: {video_path}")


def organize_output_files(output_dir):
    """Separate videos from the other generated files."""

    output_dir = Path(output_dir)
    partial_dir = output_dir / "partial"
    final_dir = output_dir / "final"
    partial_dir.mkdir(exist_ok=True)
    final_dir.mkdir(exist_ok=True)

    for file_path in output_dir.iterdir():
        if not file_path.is_file():
            continue

        destination_dir = (
            final_dir
            if file_path.suffix.lower() == ".mp4"
            else partial_dir
        )
        shutil.move(str(file_path), destination_dir / file_path.name)


def main():

    parser = argparse.ArgumentParser(
        description="Gera uma música a partir de um checkpoint treinado."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "Caminho do checkpoint .pt; "
            "padrão: output/model.pt."
        ),
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    print(f"Using checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}. "
            "Execute primeiro: python src/train.py"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    vocab = Vocabulary()

    if checkpoint["vocab_size"] != vocab.size:
        raise RuntimeError(
            "O checkpoint usa um vocabulário incompatível. "
            "Execute novamente: python src/train.py"
        )

    model = MusicTransformer(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint["dropout"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(
                f"Checkpoint inválido: {name}"
            )

    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab.size}")
    print(f"Maximum tracks: {MAX_TRACKS}")

    output_dir = create_output_directory()
    output_path = output_dir / "generated.mid"

    tokens = generate(
        model,
        vocab,
        device,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS
    )

    tokens_to_midi(
        tokens,
        vocab,
        output_path
    )

    print(f"MIDI generated: {output_path}")
    print(f"Token count: {len(tokens)}")
    generate_media_files(output_path)
    organize_output_files(output_dir)


if __name__ == "__main__":
    main()
