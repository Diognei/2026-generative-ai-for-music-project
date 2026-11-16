"""Generate a compact pianoroll figure from a MIDI file.

Usage:

    python ga/generate_pianoroll.py final_best.mid
    python ga/generate_pianoroll.py final_best.mid pianoroll.png
"""

from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, MultipleLocator
import pretty_midi


DARK_PIANOROLL = True


DEFAULT_OUTPUT = "pianoroll.png"

BACKGROUND_COLOR = "#05070a"
AXES_BACKGROUND_COLOR = "#0b1017"
TEXT_COLOR = "#e6edf3"
SPINE_COLOR = "#64748b"
GRID_MAJOR_COLOR = "#94a3b8"
GRID_MINOR_COLOR = "#64748b"
TRACK_COLORS = (
    "#00d7ff",
    "#ff6b6b",
    "#ffd166",
    "#06d6a0",
    "#c77dff",
    "#ff9f1c",
    "#a8dadc",
    "#f72585",
    "#b8f200",
    "#4cc9f0",
)


def _instrument_name(instrument, index):
    """Return a short name based on the General MIDI program."""

    if instrument.is_drum:
        return "Drums"

    program_family = {
        range(0, 8): "Piano",
        range(8, 16): "Percussion",
        range(16, 24): "Organ",
        range(24, 32): "Guitar",
        range(32, 40): "Bass",
        range(40, 48): "Strings",
        range(48, 56): "Ensemble",
        range(56, 64): "Brass",
        range(64, 72): "Reed",
        range(72, 80): "Pipe",
        range(80, 88): "Synth",
        range(88, 96): "Synth",
        range(96, 104): "Synth",
        range(104, 112): "Ethnic",
        range(112, 120): "Percussion",
        range(120, 128): "Effects",
    }

    for programs, name in program_family.items():
        if instrument.program in programs:
            return name

    return "Instrument"


def generate_pianoroll(midi_path, output_path=DEFAULT_OUTPUT, dpi=180):
    """Generate and save a compact pianoroll with one row per track."""

    midi_path = Path(midi_path)
    output_path = Path(output_path)
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    tracks = list(midi.instruments)

    if not tracks:
        raise ValueError("O arquivo MIDI não possui pistas para desenhar.")

    audible_end = max(
        (
            note.end
            for instrument in tracks
            for note in instrument.notes
            if note.velocity > 0
        ),
        default=0.0,
    )

    duration = max(
        midi.get_end_time(),
        audible_end,
    )

    # --------------------------------------------------------
    # Bar grid (x-axis in bars, not seconds)
    # --------------------------------------------------------

    # get_tempo_changes() returns (time points, tempos).
    _tempo_change_times, tempos = midi.get_tempo_changes()
    tempo = float(tempos[0]) if len(tempos) else 120.0

    time_signatures = getattr(
        midi,
        "time_signature_changes",
        []
    )

    beats_per_bar = (
        time_signatures[0].numerator
        if time_signatures
        else 4
    )

    bar_duration = (
        60.0 / tempo
    ) * beats_per_bar

    duration_bars = duration / bar_duration
    total_bars = int(math.ceil(duration_bars))

    # --------------------------------------------------------
    # Major ticks
    #
    # The markers start at bar 1 and appear every
    # 4 bars, matching the current 4-bar intro:
    #
    #     1, 5, 9, 13, 17, ...
    #
    # The last bar is also shown separately.
    # --------------------------------------------------------

    bar_ticks = list(
        range(0, max(1, total_bars - 1), 4)
    )

    if DARK_PIANOROLL:
        figure_background = BACKGROUND_COLOR
        axes_background = AXES_BACKGROUND_COLOR
        text_color = TEXT_COLOR
        spine_color = SPINE_COLOR
        grid_major_color = GRID_MAJOR_COLOR
        grid_minor_color = GRID_MINOR_COLOR
        grid_major_alpha = 0.32
        grid_minor_alpha = 0.18
        grid_major_width = 0.6
        grid_minor_width = 0.4
        track_colormap = None
    else:
        figure_background = "white"
        axes_background = "white"
        text_color = "black"
        spine_color = "black"
        grid_major_color = "#b0b0b0"
        grid_minor_color = "#b0b0b0"
        grid_major_alpha = 0.25
        grid_minor_alpha = 0.10
        grid_major_width = 0.5
        grid_minor_width = 0.4
        track_colormap = plt.get_cmap("tab20")

    figure_height = max(
        1.8,
        0.65 * len(tracks)
    )

    figure, axes = plt.subplots(
        len(tracks),
        1,
        figsize=(14, figure_height),
        sharex=True,
        squeeze=False,
        facecolor=figure_background,
    )
    figure.patch.set_facecolor(figure_background)

    for row, (instrument, axis) in enumerate(
        zip(tracks, axes[:, 0])
    ):
        axis.set_facecolor(axes_background)
        track_color = (
            TRACK_COLORS[row % len(TRACK_COLORS)]
            if DARK_PIANOROLL
            else track_colormap(row % 20)
        )

        visible_notes = [
            note
            for note in instrument.notes
            if note.velocity > 0 and note.pitch != 127
        ]

        pitches = [
            note.pitch
            for note in visible_notes
        ]

        if pitches:

            minimum_pitch = min(pitches)
            maximum_pitch = max(pitches)

        else:

            # Empty track (e.g., reserved String2/Bass2/Guitar2):
            # draw the empty row with a standard pitch range.
            minimum_pitch = 60
            maximum_pitch = 72

        pitch_padding = max(
            1,
            (maximum_pitch - minimum_pitch) * 0.08
        )

        for note in visible_notes:

            axis.add_patch(
                Rectangle(
                    (
                        note.start / bar_duration,
                        note.pitch - 0.42,
                    ),
                    max(
                        note.end - note.start,
                        0.001
                    ) / bar_duration,
                    0.84,
                    facecolor=track_color,
                    edgecolor="none",
                )
            )

        axis.set_ylabel(
            _instrument_name(instrument, row),
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
            fontsize=8,
            color=text_color,
        )

        axis.set_ylim(
            minimum_pitch - pitch_padding,
            maximum_pitch + pitch_padding,
        )

        # ----------------------------------------------------
        # Major ticks:
        #
        # 1, 5, 9, 13, ...
        # ----------------------------------------------------

        axis.xaxis.set_major_locator(
            FixedLocator(bar_ticks)
        )

        # ----------------------------------------------------
        # Minor ticks:
        #
        # Keep one marker per bar.
        # ----------------------------------------------------

        axis.xaxis.set_minor_locator(
            MultipleLocator(1)
        )

        axis.tick_params(
            axis="x",
            which="major",
            labelsize=7,
            length=3,
            colors=text_color,
        )

        axis.tick_params(
            axis="x",
            which="minor",
            length=2,
            colors=spine_color,
        )

        axis.grid(
            axis="x",
            which="major",
            color=grid_major_color,
            alpha=grid_major_alpha,
            linewidth=grid_major_width,
        )

        axis.grid(
            axis="x",
            which="minor",
            color=grid_minor_color,
            alpha=grid_minor_alpha,
            linewidth=grid_minor_width,
        )

        axis.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
            colors=text_color,
        )

        for spine in axis.spines.values():
            spine.set_color(spine_color)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # Only the bottom axis shows labels:
    # the bar number.
    # --------------------------------------------------------

    for axis in axes[:-1, 0]:

        axis.tick_params(
            axis="x",
            labelbottom=False,
            colors=text_color,
        )

    axes[-1, 0].set_xlim(
        0,
        duration_bars
    )

    axes[-1, 0].set_xticks(
        bar_ticks
    )

    # The axis value represents:
    #
    #   0 = start of bar 1
    #   4 = start of bar 5
    #   8 = start of bar 9
    #
    # Therefore, add 1 to display the actual
    # bar number.

    axes[-1, 0].set_xticklabels(
        [
            str(tick + 1)
            for tick in bar_ticks
        ],
        fontsize=7,
        color=text_color,
    )

    axes[-1, 0].set_xlabel(
        "Compass",
        color=text_color,
    )

    figure.suptitle(
        midi_path.name,
        fontsize=11,
        color=text_color,
    )

    figure.subplots_adjust(
        left=0.16,
        right=0.99,
        top=0.90,
        bottom=0.12,
        hspace=0.18,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        edgecolor=figure.get_facecolor(),
    )

    plt.close(figure)

    return output_path


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Gera um pianoroll compacto "
            "a partir de um arquivo MIDI."
        )
    )

    parser.add_argument(
        "midi_path",
        type=Path,
        help="Caminho do arquivo MIDI de entrada.",
    )

    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        default=Path(DEFAULT_OUTPUT),
        help=(
            "Caminho da figura de saída; "
            "padrão: pianoroll.png."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Resolução da figura; padrão: 180.",
    )

    args = parser.parse_args()

    output_path = generate_pianoroll(
        args.midi_path,
        args.output_path,
        dpi=args.dpi,
    )

    print(
        f"Pianoroll saved to: {output_path}"
    )


if __name__ == "__main__":

    main()
