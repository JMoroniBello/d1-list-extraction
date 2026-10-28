"""Portable, pickle-free snapshots. They contain source text and must be handled as source data."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from .questions import KINDS


@dataclass
class Snapshot:
    text: str
    logits: np.ndarray
    offsets: np.ndarray
    native_answers: dict
    kinds: tuple[str, ...] = KINDS
    questions: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.kinds = tuple(self.kinds)
        self.logits = np.array(self.logits, dtype=np.float32, copy=True)
        self.offsets = np.array(self.offsets, dtype=np.int64, copy=True)
        if not isinstance(self.text, str) or set(self.kinds) != set(KINDS) or len(self.kinds) != 4:
            raise ValueError("A snapshot needs source text and the four standard entity question kinds")
        if self.logits.ndim != 3 or self.logits.shape[:2] != (4, 2):
            raise ValueError("Expected logits shaped [4, 2, source_positions]")
        if self.offsets.shape != (self.logits.shape[2], 2) or len(self.offsets) == 0:
            raise ValueError("Offsets must match source positions")
        if np.isnan(self.logits).any() or np.isposinf(self.logits).any() or not np.isfinite(self.logits[:, :, 0]).all():
            raise ValueError("Invalid scores or null position")
        if self.offsets[0].tolist() != [-1, -1]:
            raise ValueError("Position zero must be the native null position")
        for start, end in self.offsets:
            if [int(start), int(end)] != [-1, -1] and not 0 <= start < end <= len(self.text):
                raise ValueError("Invalid source offsets")
        if set(self.native_answers) != set(self.kinds):
            raise ValueError("Native answers must match question kinds")
        for answer in self.native_answers.values():
            if answer.get("text") is not None:
                span = answer.get("span")
                if not isinstance(span, list) or len(span) != 2 or not 0 <= span[0] < span[1] <= len(self.text):
                    raise ValueError("Invalid native answer span")
                if self.text[slice(*span)] != answer["text"]:
                    raise ValueError("Native answer differs from source")
        self.logits.setflags(write=False)
        self.offsets.setflags(write=False)

    def save(self, path):
        meta = {"format": "d1-list-scores-v1", "text": self.text, "kinds": self.kinds,
                "native_answers": self.native_answers, "questions": self.questions, "metadata": self.metadata}
        # File handle prevents numpy silently appending another extension.
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, logits=self.logits, offsets=self.offsets,
                                metadata=np.array(json.dumps(meta, ensure_ascii=False)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            meta = json.loads(str(arrays["metadata"].item()))
            if meta.pop("format") != "d1-list-scores-v1":
                raise ValueError("Unsupported snapshot format")
            return cls(logits=arrays["logits"], offsets=arrays["offsets"], **meta)

