
class Vocabulary:
    """Vocabulary for a globally timed, multi-track MIDI event stream."""

    MAX_TRACKS = 32

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

        def add(token):
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token

        for token in [
            "PAD", "MUSIC_START", "MUSIC_END", "TRACK_START",
            "TRACK_END", "IS_DRUM", "IS_MELODIC"
        ]:
            add(token)

        for track in range(32):
            add(f"TRACK_SELECT_{track}")

        for i in range(128):
            add(f"PROGRAM_{i}")

        for i in range(128):
            add(f"NOTE_ON_{i}")
            add(f"NOTE_OFF_{i}")

        for i in range(1, 128):
            add(f"VELOCITY_{i}")

        for i in range(1, 101):
            add(f"TIME_SHIFT_{i}")

        self.size = len(self.token_to_id)

        self.track_select_ids = self.ids("TRACK_SELECT", 32)
        self.program_ids = self.ids("PROGRAM", 128)
        self.note_on_ids = [self.encode(f"NOTE_ON_{i}") for i in range(128)]
        self.note_off_ids = [self.encode(f"NOTE_OFF_{i}") for i in range(128)]
        self.velocity_ids = self.ids("VELOCITY", 127, start=1)
        self.time_shift_ids = self.ids("TIME_SHIFT", 100, start=1)

    def ids(self, prefix, count, start=0):
        return [
            self.token_to_id[f"{prefix}_{i}"]
            for i in range(start, start + count)
        ]

    def encode(self, token):
        return self.token_to_id[token]

    def decode(self, idx):
        return self.id_to_token[int(idx)]

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2)

    @property
    def pad_id(self):
        return self.encode("PAD")

    @property
    def music_start_id(self):
        return self.encode("MUSIC_START")

    @property
    def music_end_id(self):
        return self.encode("MUSIC_END")

    @property
    def track_start_id(self):
        return self.encode("TRACK_START")

    @property
    def track_end_id(self):
        return self.encode("TRACK_END")

    @property
    def is_drum_id(self):
        return self.encode("IS_DRUM")

    @property
    def is_melodic_id(self):
        return self.encode("IS_MELODIC")
