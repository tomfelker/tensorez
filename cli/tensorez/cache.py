"""Stage result caching (CONTRACT.md §3).

Layout: ``cache/<stage>/<sha256(key)[:16]>/`` containing the stage's saved
arrays plus ``key.txt``, a human-readable dump of everything that went into
the key: input files identified by path + size + mtime, all parameters that
affect the stage, and the keys of upstream stages (so invalidation chains).

``key.txt`` is written *last* and its presence marks the entry complete;
a killed run leaves at worst an incomplete entry that is simply recomputed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

KEY_FILENAME = "key.txt"


class CacheEntry:
    def __init__(self, cache_dir: Path, stage: str, key_text: str):
        self.key_text = key_text
        self.key_hash = hashlib.sha256(key_text.encode("utf-8")).hexdigest()[:16]
        self.dir = Path(cache_dir) / stage / self.key_hash

    @property
    def complete(self) -> bool:
        return (self.dir / KEY_FILENAME).is_file()

    def load_npz(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.dir / f"{name}.npz") as data:
            return {k: data[k] for k in data.files}

    def save_npz(self, name: str, **arrays: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{name}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, self.dir / f"{name}.npz")

    def mark_complete(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / (KEY_FILENAME + ".tmp")
        tmp.write_text(self.key_text)
        os.replace(tmp, self.dir / KEY_FILENAME)
