
class ConstrainedDecoder:
    """Control the structure of MIDI sequences during generation."""

    def __init__(
        self,
        vocab,
        max_tracks=16,
        max_polyphony=32,
        track_plan=None,
        max_track_duration=None,
        track_budgets=None,
        max_note_duration_steps=None,
    ):
        self.vocab = vocab
        self.max_tracks = max_tracks
        self.max_polyphony = max_polyphony
        self.track_plan = track_plan
        self.track_budgets = track_budgets
        self.max_time_steps = (
            None
            if max_track_duration is None
            else max(1, round(max_track_duration / 0.01))
        )
        self.max_note_duration_steps = max_note_duration_steps

    def _state(self, sequence):
        v = self.vocab

        state = "need_track"
        active = set()
        pending_velocity = False
        note_count = 0
        tracks = 0
        current_time_steps = 0
        current_track_tokens = 0
        note_start_times = {}

        for idx in sequence[1:]:
            token = v.decode(idx)

            if state == "need_track":

                if token == "TRACK_START":
                    state = "need_track_type"
                    current_track_tokens = 1

            elif state == "need_track_type":
                current_track_tokens += 1

                if token == "IS_DRUM":
                    state = "events"

                elif token == "IS_MELODIC":
                    state = "need_program"

            elif state == "need_program":
                current_track_tokens += 1

                if token.startswith("PROGRAM_"):
                    state = "events"

            elif state == "events":
                current_track_tokens += 1

                if token.startswith("TIME_SHIFT_"):
                    current_time_steps += int(token.split("_")[2])

                elif token.startswith("VELOCITY_"):
                    pending_velocity = True

                elif token.startswith("NOTE_ON_"):

                    pitch = int(token.split("_")[2])

                    active.add(pitch)
                    note_start_times[pitch] = current_time_steps
                    pending_velocity = False
                    note_count += 1

                elif token.startswith("NOTE_OFF_"):

                    pitch = int(token.split("_")[2])
                    active.discard(pitch)
                    note_start_times.pop(pitch, None)

                elif token == "TRACK_END":

                    active.clear()
                    note_start_times.clear()
                    tracks += 1
                    state = "after_track"

            elif state == "after_track":

                if token == "TRACK_START":
                    current_time_steps = 0
                    note_start_times.clear()
                    state = "need_track_type"
                    current_track_tokens = 1

                elif token == "MUSIC_END":
                    state = "ended"

        return (
            state,
            active,
            pending_velocity,
            note_count,
            tracks,
            current_time_steps,
            current_track_tokens,
            note_start_times,
        )

    def allowed_next(self, sequence):
        v = self.vocab

        (
            state,
            active,
            pending,
            note_count,
            tracks,
            current_time_steps,
            current_track_tokens,
            note_start_times,
        ) = (
            self._state(sequence)
        )

        if state == "need_track":
            return [v.track_start_id]

        if state == "need_track_type":

            if self.track_plan is not None and tracks < len(self.track_plan):
                is_drum, _ = self.track_plan[tracks]
                return [v.is_drum_id if is_drum else v.is_melodic_id]

            return [
                v.is_drum_id,
                v.is_melodic_id
            ]

        if state == "need_program":

            if self.track_plan is not None and tracks < len(self.track_plan):
                _, program = self.track_plan[tracks]
                return [v.encode(f"PROGRAM_{program}")]

            return v.program_ids

        if state == "events":

            # Notes that have exceeded the maximum duration.
            expired_pitches = set()
            if self.max_note_duration_steps is not None:
                for pitch in list(active):
                    started = note_start_times.get(pitch, 0)
                    if current_time_steps - started >= self.max_note_duration_steps:
                        expired_pitches.add(pitch)

            # If a note has expired and velocity is not pending,
            # only allow those notes to be turned off now.
            if expired_pitches and not pending:
                return [
                    i for i in v.note_off_ids
                    if int(v.decode(i).split("_")[2]) in expired_pitches
                ]

            budget_reached = (
                self.track_budgets is not None
                and tracks < len(self.track_budgets)
                and current_track_tokens >= self.track_budgets[tracks]
            )

            if pending:
                return [
                    i for i in v.note_on_ids
                    if int(v.decode(i).split("_")[2]) not in active
                ]

            if budget_reached:
                allowed = []
            elif self.max_time_steps is None:
                allowed = list(v.time_shift_ids)
            else:
                remaining = self.max_time_steps - current_time_steps
                allowed = [
                    token_id
                    for token_id in v.time_shift_ids
                    if int(v.decode(token_id).split("_")[2]) <= remaining
                ]

            if not budget_reached and len(active) < self.max_polyphony:
                allowed.extend(v.velocity_ids)

            allowed.extend(
                i for i in v.note_off_ids
                if int(v.decode(i).split("_")[2]) in active
            )

            if not active and note_count > 0:
                allowed.append(v.track_end_id)

            return allowed

        if state == "after_track":

            allowed = []

            if self.track_plan is not None:
                if tracks < len(self.track_plan):
                    allowed.append(v.track_start_id)
                else:
                    allowed.append(v.music_end_id)
            elif tracks < self.max_tracks:
                allowed.extend([v.music_end_id, v.track_start_id])
            else:
                allowed.append(v.music_end_id)

            return allowed

        if state == "ended":
            return [v.music_end_id]

        return [v.music_end_id]

    def force_close(self, sequence):
        """Finish a sequence when the token limit is reached."""

        v = self.vocab

        (
            state,
            active,
            pending,
            note_count,
            tracks,
            current_time_steps,
            current_track_tokens,
            note_start_times,
        ) = (
            self._state(sequence)
        )

        if state == "need_track":

            sequence.append(v.track_start_id)
            state = "need_track_type"

        if state == "need_track_type":

            sequence.append(v.is_melodic_id)
            sequence.append(v.encode("PROGRAM_0"))

            state = "events"
            pending = False
            note_count = 0

        if state == "need_program":

            sequence.append(v.encode("PROGRAM_0"))
            state = "events"

        if state == "events":

            if pending:
                free_pitch = next(
                    p for p in range(128)
                    if p not in active
                )

                sequence.append(
                    v.encode(f"NOTE_ON_{free_pitch}")
                )

                active.add(free_pitch)
                note_count += 1

            if note_count == 0:

                sequence.extend([
                    v.encode("VELOCITY_80"),
                    v.encode("NOTE_ON_60")
                ])

                active.add(60)

            for pitch in sorted(active):
                sequence.append(
                    v.encode(f"NOTE_OFF_{pitch}")
                )

            sequence.append(v.track_end_id)
            state = "after_track"
            tracks += 1

        if state == "after_track" and self.track_plan is not None:
            while tracks < len(self.track_plan):
                is_drum, program = self.track_plan[tracks]
                sequence.append(v.track_start_id)
                sequence.append(
                    v.is_drum_id if is_drum else v.is_melodic_id
                )

                if not is_drum:
                    sequence.append(v.encode(f"PROGRAM_{program}"))

                sequence.extend([
                    v.encode("VELOCITY_80"),
                    v.encode("NOTE_ON_60"),
                    v.encode("NOTE_OFF_60"),
                    v.track_end_id
                ])
                tracks += 1

        if state == "after_track":
            sequence.append(v.music_end_id)

        return sequence


class ConstrainedDecoder:
    """Constrains a single global clock while allowing events on every track."""

    def __init__(self, vocab, max_tracks=32, max_polyphony=32, track_plan=None,
                 max_track_duration=None, track_budgets=None,
                 max_note_duration_steps=None):
        self.vocab = vocab
        self.max_tracks = max_tracks
        self.max_polyphony = max_polyphony
        self.track_count = len(track_plan or [])
        self.max_time_steps = None if max_track_duration is None else max(1, round(max_track_duration / 0.01))
        self.max_note_duration_steps = max_note_duration_steps

    def _state(self, sequence):
        current_time = 0
        selected_track = None
        active = set()
        pending_velocity = False
        note_count = 0
        for token_id in sequence[1:]:
            token = self.vocab.decode(token_id)
            if token.startswith("TIME_SHIFT_"):
                current_time += int(token.split("_")[2])
            elif token.startswith("TRACK_SELECT_"):
                selected_track = int(token.split("_")[2])
            elif token.startswith("VELOCITY_"):
                pending_velocity = True
            elif token.startswith("NOTE_ON_") and selected_track is not None:
                pitch = int(token.split("_")[2])
                active.add((selected_track, pitch))
                pending_velocity = False
                note_count += 1
            elif token.startswith("NOTE_OFF_") and selected_track is not None:
                active.discard((selected_track, int(token.split("_")[2])))
        return current_time, selected_track, active, pending_velocity, note_count

    def allowed_next(self, sequence):
        current_time, selected_track, active, pending, note_count = self._state(sequence)
        if pending and selected_track is not None:
            return [token_id for token_id in self.vocab.note_on_ids
                    if (selected_track, int(self.vocab.decode(token_id).split("_")[2])) not in active]

        allowed = list(self.vocab.track_select_ids[:self.track_count or self.max_tracks])
        if self.max_time_steps is None:
            allowed.extend(self.vocab.time_shift_ids)
        else:
            remaining = self.max_time_steps - current_time
            allowed.extend(token_id for token_id in self.vocab.time_shift_ids
                           if int(self.vocab.decode(token_id).split("_")[2]) <= remaining)
        selected_track_notes = {
            pitch for track, pitch in active
            if track == selected_track
        }
        if len(selected_track_notes) < self.max_polyphony:
            allowed.extend(self.vocab.velocity_ids)
        allowed.extend(token_id for token_id in self.vocab.note_off_ids
                       if selected_track is not None and
                       (selected_track, int(self.vocab.decode(token_id).split("_")[2])) in active)
        if note_count > 0 and not active:
            allowed.append(self.vocab.music_end_id)
        return allowed

    def force_close(self, sequence):
        _, _, active, _, _ = self._state(sequence)
        for track, pitch in sorted(active):
            sequence.extend([
                self.vocab.encode(f"TRACK_SELECT_{track}"),
                self.vocab.encode(f"NOTE_OFF_{pitch}")
            ])
        sequence.append(self.vocab.music_end_id)
        return sequence
