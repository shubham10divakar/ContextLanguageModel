"""ContextBench: diagnostic tasks that isolate context management (paper Sec. 3, App. D).

Levels follow Fig. 15: each task reaches higher context pressure through one
generator setting, everything else fixed.
"""

from .common import READY, StreamTask
from .kv_store import KVStore
from .log_triage import LogTriage
from .needle import NeedleRetention
from .sudoku import SudokuSketchpad

TASKS = {
    "needle_retention": (NeedleRetention, "n_chunks"),
    "sudoku_sketchpad": (SudokuSketchpad, "n_moves"),
    "kv_store": (KVStore, "n_records"),
    "log_triage": (LogTriage, "n_lines"),
}

# Generator settings per pressure level, read off Fig. 15.
LEVELS = {
    "needle_retention": [4, 8, 24, 48, 60, 96, 192, 224],
    "sudoku_sketchpad": [4, 8, 15, 30, 60, 100, 150],
    "kv_store": [130, 260, 520, 1000, 2000, 4000, 8000, 16000],
    "log_triage": [160, 320, 640, 1600, 3200, 12800],
}


def make_task(name: str, level: int, seed: int = 0, **kwargs) -> StreamTask:
    cls, knob = TASKS[name]
    return cls(**{knob: level}, seed=seed, **kwargs)


__all__ = ["READY", "TASKS", "LEVELS", "make_task", "StreamTask", "KVStore", "LogTriage",
           "NeedleRetention", "SudokuSketchpad"]
