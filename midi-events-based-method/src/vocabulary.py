
class Vocabulary:
    """Vocabulário fixo para representação MIDI baseada em eventos."""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

        def add(token):
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        special_tokens = [
            "PAD",
            "MUSIC_START",
            "MUSIC_END",
            "TRACK_START",
            "TRACK_END",
            "IS_DRUM",
            "IS_MELODIC"
        ]

        for token in special_tokens:
            add(token)

        for i in range(128):
            add(f"PROGRAM_{i}")

        for i in range(128):
            add(f"NOTE_ON_{i}")

        for i in range(128):
            add(f"NOTE_OFF_{i}")

        for i in range(1, 128):
            add(f"VELOCITY_{i}")

        for i in range(1, 101):
            add(f"TIME_SHIFT_{i}")

        self.size = len(self.token_to_id)

        self.program_ids = self.ids("PROGRAM", 128)
        self.note_on_ids = self.ids("NOTE_ON", 128)
        self.note_off_ids = self.ids("NOTE_OFF", 128)
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
        return self.token_to_id["PAD"]

    @property
    def music_start_id(self):
        return self.token_to_id["MUSIC_START"]

    @property
    def music_end_id(self):
        return self.token_to_id["MUSIC_END"]

    @property
    def track_start_id(self):
        return self.token_to_id["TRACK_START"]

    @property
    def track_end_id(self):
        return self.token_to_id["TRACK_END"]

    @property
    def is_drum_id(self):
        return self.token_to_id["IS_DRUM"]

    @property
    def is_melodic_id(self):
        return self.token_to_id["IS_MELODIC"]
