"""Sudoku Sketchpad: surgical in-place updates.

A 16x16 board is kept as a sketchpad of ``(row,col,value)`` tuples. Users
stream one move per operation; after each move the most recent SKETCHPAD
block in the agent's context must equal the expected board with the right
VERSION. Score = fraction of moves reproduced exactly.
"""

from __future__ import annotations

import re

from .common import Op, StreamTask

SYMS = "123456789ABCDEFG"
SIDE = 16
BASE = 4
BLOCK_RE = re.compile(r"<<<SKETCHPAD BEGIN>>>[ \t]*\r?\nVERSION:[ \t]*(\d+)[ \t]*\r?\n(.*?)<<<SKETCHPAD END>>>", re.S)
CELL_RE = re.compile(r"\((\d+),(\d+),([^)\s]+)\)")


def solved_board(rng) -> list[list[str]]:
    def shuffled(seq):
        seq = list(seq)
        rng.shuffle(seq)
        return seq

    bands = shuffled(range(BASE))
    rows = [g * BASE + r for g in bands for r in shuffled(range(BASE))]
    stacks = shuffled(range(BASE))
    cols = [g * BASE + c for g in stacks for c in shuffled(range(BASE))]
    syms = shuffled(SYMS)
    return [[syms[(BASE * (r % BASE) + r // BASE + c) % SIDE] for c in cols] for r in rows]


def render_board(board: list[list[str]], version: int) -> str:
    rows = [" ".join(f"({r + 1},{c + 1},{board[r][c]})" for c in range(SIDE)) for r in range(SIDE)]
    return "<<<SKETCHPAD BEGIN>>>\nVERSION: {}\n{}\n<<<SKETCHPAD END>>>".format(version, "\n".join(rows))


def parse_last_board(text: str) -> tuple[int, dict] | None:
    blocks = list(BLOCK_RE.finditer(text))
    if not blocks:
        return None
    m = blocks[-1]
    cells = {(int(r), int(c)): v for r, c, v in CELL_RE.findall(m.group(2))}
    return int(m.group(1)), cells


class SudokuSketchpad(StreamTask):
    name = "sudoku_sketchpad"
    header_instruction = (
        "TASK: Sudoku Sketchpad. You maintain a 16x16 sketchpad of (row,col,value) tuples, where '.' is an empty "
        "cell, wrapped in <<<SKETCHPAD BEGIN>>> / <<<SKETCHPAD END>>> with a VERSION line. Each operation is one "
        "move. After you handle a move, the most recent SKETCHPAD block in your context must be the updated board "
        "with the new VERSION, in exactly the same format; that block is what gets graded for that move.")

    def __init__(self, n_moves: int = 4, fill_ratio: float = 0.4, seed: int = 0):
        self.n_moves = n_moves
        self.fill_ratio = fill_ratio
        super().__init__(seed)

    def generate(self) -> list[Op]:
        rng = self.rng
        solution = solved_board(rng)
        cells = [(r, c) for r in range(SIDE) for c in range(SIDE)]
        rng.shuffle(cells)
        n_fill = int(self.fill_ratio * len(cells))
        empties = cells[n_fill:]
        if self.n_moves > len(empties):
            raise ValueError("too many moves for the number of empty cells")
        board = [["." for _ in range(SIDE)] for _ in range(SIDE)]
        for r, c in cells[:n_fill]:
            board[r][c] = solution[r][c]
        self.boards = [[row[:] for row in board]]
        ops = []
        for k, (r, c) in enumerate(empties[: self.n_moves], 1):
            v = solution[r][c]
            board[r][c] = v
            self.boards.append([row[:] for row in board])
            body = ""
            if k == 1:
                body += "Your STARTING sketchpad (version 0):\n" + render_board(self.boards[0], 0) + "\n\n"
            body += (f"Move (board #{k}): Place {v} in cell r{r + 1}c{c + 1} (row {r + 1} from the top, column "
                     f"{c + 1} from the left).\nIn your sketchpad, set that cell's tuple value to {v} and set the "
                     f"VERSION line to {k}. Leave every other cell unchanged.")
            ops.append(Op(title=f"board 1 move v{k}", body=body, kind="move", payload={"r": r, "c": c, "v": v}))
        return ops

    def is_graded(self, op: Op) -> bool:
        return True

    def total_input_tokens(self, counter=None) -> int:
        # Fig. 15 counts the board once per move for Sudoku.
        from ..tokens import TokenCounter

        counter = counter or TokenCounter()
        board_tokens = counter.count_text(render_board(self.boards[-1], self.n_moves))
        return super().total_input_tokens(counter) + board_tokens * self.n_moves

    def score_op(self, i: int, context: str) -> float:
        parsed = parse_last_board(context)
        if parsed is None:
            return 0.0
        version, cells = parsed
        expected = self.boards[i + 1]
        want = {(r + 1, c + 1): expected[r][c] for r in range(SIDE) for c in range(SIDE)}
        return float(version == i + 1 and cells == want)
