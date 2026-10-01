"""Scripted reference policies for ContextBench.

``clm_oracle`` behaves like an ideal CLM: every turn it edits the context file
with a Python script (offload, prune, rewrite in place) and answers from disk.
It should score 100% at every level; if it does not, the grader or harness is
broken. ``keep_all_oracle`` is an ideal reader that never manages context: it
answers perfectly from what is in context, so its failures under the ``base``
strategy come purely from overflow. Together they bracket any real model.
"""

from __future__ import annotations

import re

from .common import READY

PRE = r'''python3 - <<'EOF'
import os, re, glob
p = os.environ["CLM_CTX"]
s = open(p, encoding="utf-8").read()
'''
POST = r'''
open(p, "w", encoding="utf-8").write("[[CTX_TURN 1 role=notes]]\n" + NOTE + "\n")
print("READY_FOR_NEXT_OP")
EOF'''


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m.get("content") or ""
    return ""


def _script(body: str) -> str:
    return PRE + body + POST


# --------------------------------------------------------------------- CLM oracle
NEEDLE = r'''
needles = []
for line in s.splitlines():
    if re.match(r"^\[n\d{5}i\d{2}#[0-9a-f]{8}\] ", line) and line not in needles:
        needles.append(line)
NOTE = "NEEDLES KEPT:\n" + "\n".join(needles)
'''

KV_SET = r'''
os.makedirs("kv", exist_ok=True)
for m in re.finditer(r"<<<SET-BATCH (\S+) BEGIN[^\n]*\n(.*?)\n<<<SET-BATCH \1 END>>>", s, re.S):
    open("kv/batch_" + m.group(1).replace("#", "_") + ".txt", "w", encoding="utf-8").write(m.group(2) + "\n")
NOTE = "KV records are offloaded to ./kv/*.txt (lookup: grep '^SET <key> = ' kv/*.txt)"
'''

KV_GET = r'''
key = "{key}"
val = None
for f in sorted(glob.glob("kv/*.txt")):
    for line in open(f, encoding="utf-8"):
        if line.startswith("SET " + key + " = "):
            val = line[len("SET " + key + " = "):].rstrip("\n")
NOTE = "KV records are offloaded to ./kv/*.txt (lookup: grep '^SET <key> = ' kv/*.txt)"
print("<<<ANSWER key=" + key + ">>>")
print(val)
print("<<<ANSWER END>>>")
'''

LOG_BATCH = r'''
os.makedirs("logs", exist_ok=True)
with open("logs/all.txt", "a", encoding="utf-8") as f:
    for m in re.finditer(r"<<<LOG-BATCH (\S+) BEGIN[^\n]*\n(.*?)\n<<<LOG-BATCH \1 END>>>", s, re.S):
        f.write(m.group(2) + "\n")
NOTE = "Log lines are offloaded to ./logs/all.txt"
'''

LOG_QUERY = r'''
lines = open("logs/all.txt", encoding="utf-8").read().splitlines()
{expr}
NOTE = "Log lines are offloaded to ./logs/all.txt"
print("<<<ANSWER qid={qid}>>>")
print(ans)
print("<<<ANSWER END>>>")
'''

SUDOKU = r'''
B = "<<<SKETCHPAD " + "BEGIN>>>"
E = "<<<SKETCHPAD " + "END>>>"
ver, body = re.findall(re.escape(B) + r"[ \t]*\nVERSION:[ \t]*(\d+)[ \t]*\n(.*?)" + re.escape(E), s, re.S)[-1]
grid = {{(int(a), int(b)): v for a, b, v in re.findall(r"\((\d+),(\d+),([^)\s]+)\)", body)}}
grid[({r}, {c})] = "{v}"
rows = [" ".join("(%d,%d,%s)" % (i, j, grid[(i, j)]) for j in range(1, 17)) for i in range(1, 17)]
NOTE = B + "\nVERSION: {k}\n" + "\n".join(rows) + "\n" + E
'''

Q_SVC = re.compile(r'How many \[(\w+)\] log lines are from service "([\w-]+)"\?')
Q_TOT = re.compile(r"How many \[(\w+)\] log lines are there in total\?")
Q_REQ = re.compile(r"request id req=([0-9a-f]+)\?")
MOVE = re.compile(r"Place (\S+) in cell r(\d+)c(\d+).*?VERSION line to (\d+)", re.S)


def _log_expr(op: str) -> str:
    if m := Q_SVC.search(op):
        return f'ans = sum(1 for l in lines if "[{m.group(1)}] {m.group(2)} " in l)'
    if m := Q_TOT.search(op):
        return f'ans = sum(1 for l in lines if "[{m.group(1)}] " in l)'
    if m := Q_REQ.search(op):
        return f'ans = next(l.split()[2] for l in lines if "req={m.group(1)} " in l)'
    raise ValueError(f"unknown query: {op[:200]}")


def clm_oracle(task_name: str):
    def policy(messages):
        op = _last_user(messages)
        if task_name == "needle_retention":
            return _script(NEEDLE)
        if task_name == "kv_store":
            if m := re.search(r"^GET (K\d+)", op, re.M):
                return _script(KV_GET.replace("{key}", m.group(1)))
            return _script(KV_SET)
        if task_name == "log_triage":
            if m := re.search(r"^QUERY (\d+):", op, re.M):
                return _script(LOG_QUERY.replace("{expr}", _log_expr(op)).replace("{qid}", m.group(1)))
            return _script(LOG_BATCH)
        if task_name == "sudoku_sketchpad":
            v, r, c, k = MOVE.search(op).groups()
            return _script(SUDOKU.format(r=r, c=c, v=v, k=k))
        raise ValueError(task_name)

    return policy


# --------------------------------------------------------------- keep-all oracle
def keep_all_oracle(task_name: str):
    """Never edits context; answers perfectly from whatever is in context."""

    def policy(messages):
        op = _last_user(messages)
        text = "\n".join(m.get("content") or "" for m in messages)
        if task_name == "kv_store" and (m := re.search(r"^GET (K\d+)", op, re.M)):
            key = m.group(1)
            vals = re.findall(rf"^SET {key} = (.*)$", text, re.M)
            val = vals[-1] if vals else "UNKNOWN"
            return f"printf '%s\\n' '<<<ANSWER key={key}>>>' '{val}' '<<<ANSWER END>>>'; echo {READY}"
        if task_name == "log_triage" and (m := re.search(r"^QUERY (\d+):", op, re.M)):
            lines = re.findall(r"^\d{4}-\d\d-\d\dT\S+ \[\w+\] .*$", text, re.M)
            ns = {"lines": lines}
            exec(_log_expr(op), ns)  # noqa: S102 - our own generated expression
            return f"printf '%s\\n' '<<<ANSWER qid={m.group(1)}>>>' '{ns['ans']}' '<<<ANSWER END>>>'; echo {READY}"
        if task_name == "sudoku_sketchpad":
            from .sudoku import parse_last_board

            v, r, c, k = MOVE.search(op).groups()
            parsed = parse_last_board(text)
            cells = parsed[1] if parsed else {}
            cells[(int(r), int(c))] = v
            rows = [" ".join(f"({i},{j},{cells.get((i, j), '.')})" for j in range(1, 17)) for i in range(1, 17)]
            board = "<<<SKETCHPAD BEGIN>>>\nVERSION: {}\n{}\n<<<SKETCHPAD END>>>".format(k, "\n".join(rows))
            return f"cat <<'EOB'\n{board}\nEOB\necho {READY}"
        return f"echo {READY}"

    return policy
