"""The path model: the likeliest place for every second, and how sure it is.

S[t, j] is how much the frame of second t looks like reference place j. Each second's row
becomes z-scores, and a Viterbi path picks one place a second: a place scores its z-score
plus the better of its best neighbour's score minus the step's cost (metres times STEP_COST)
and the best score anywhere minus JUMP. A centred moving average over WINDOW seconds then
smooths the path's positions, never across a jump. These are the lab's sixth test's settings,
chosen leave-one-raid-out on three Labs raids and checked on four raids none was tuned on.
"""
from __future__ import annotations

import numpy as np

STEP = 1.0  # seconds between frames
RADIUS = 15.0  # metres: a position this close to the truth (on the same floor) counts as right
FLOOR = 3.0  # metres of height within which two places are on the same floor
REACH = 8.0 * STEP + 10.0  # a move within this many metres in a second is a step, a longer one a jump
STEP_COST = 0.4  # per metre stepped in a second, in the z-scores the similarities are measured in
JUMP = 32.0
WINDOW = 9  # seconds
FAR = 30.0  # sureness compares the path with the best path at least this far from it
SURE = (10.0, 20.0)  # sureness tiers: unsure below the first, sure from the second
CHUNK = 1024  # places compared at once when finding neighbours (memory: places x CHUNK distances)


def zscores(S: np.ndarray) -> np.ndarray:
    """Every row as z-scores, row by row as the lab computed them (float32 sums depend on the order)."""
    Z = np.empty_like(S)
    for t, s in enumerate(S):
        Z[t] = (s - s.mean()) / max(s.std(), 1e-6)
    return Z


def neighbours(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every place j, the places a path can step to j from (within REACH, on the same floor),
    in index order, and how far they are; -1 pads the rows."""
    n = len(pos)
    rows: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    for lo in range(0, n, CHUNK):
        part = pos[lo:lo + CHUNK]
        d = np.hypot(pos[:, None, 0] - part[None, :, 0], pos[:, None, 2] - part[None, :, 2])
        local = (d <= REACH) & (np.abs(pos[:, None, 1] - part[None, :, 1]) <= FLOOR)
        for j in range(len(part)):
            idx = np.flatnonzero(local[:, j])
            rows.append(idx)
            dists.append(d[idx, j])
    width = max((len(r) for r in rows), default=1)
    nbr = np.full((n, width), -1)
    dist = np.zeros((n, width), np.float32)
    for j, (idx, d) in enumerate(zip(rows, dists)):
        nbr[j, :len(idx)] = idx
        dist[j, :len(idx)] = d
    return nbr, dist


def best_path(Z: np.ndarray, nbr: np.ndarray, cost: np.ndarray, jump: float) -> np.ndarray:
    """The best-scoring place for every second (Viterbi over steps and jumps)."""
    n = Z.shape[1]
    rows = np.arange(n)
    src = np.maximum(nbr, 0)
    cost = np.where(nbr >= 0, cost, np.inf).astype(np.float32)  # padding is never the best neighbour
    jump = np.float32(jump)
    score = Z[0]
    back = np.empty((len(Z), n), np.int32)
    for t in range(1, len(Z)):
        cand = score[src] - cost
        k = cand.argmax(1)
        best_local = cand[rows, k]
        g = int(np.argmax(score))
        use_local = best_local >= score[g] - jump
        back[t] = np.where(use_local, nbr[rows, k], g)
        score = Z[t] + np.where(use_local, best_local, score[g] - jump)
    path = [int(np.argmax(score))]
    for t in range(len(Z) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return np.array(path[::-1])


def smooth(est: np.ndarray, window: int, values: np.ndarray | None = None) -> np.ndarray:
    """Centred moving average over `window` seconds that never crosses a jump of the path `est`:
    of its positions, or of `values` (one per second) when given."""
    data = est if values is None else values.reshape(len(est), -1)
    if window == 1:
        return est if values is None else values
    out = np.empty(data.shape)
    cuts = np.flatnonzero(np.hypot(np.diff(est[:, 0]), np.diff(est[:, 2])) > REACH) + 1
    for part, o in zip(np.split(data, cuts), np.split(out, cuts)):
        csum = np.concatenate([np.zeros((1, data.shape[1])), np.cumsum(part, 0)])
        i = np.arange(len(part))
        lo = np.maximum(i - window // 2, 0)
        hi = np.minimum(i + window // 2 + 1, len(part))
        o[:] = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    return out if values is None else out.reshape(values.shape)


def sureness(Z: np.ndarray, nbr: np.ndarray, cost: np.ndarray, jump: float, pos: np.ndarray,
             path: np.ndarray, window: int) -> np.ndarray:
    """How sure the path is each second: how much lower the best path scores that is more than FAR
    metres from it (or on another floor) at that second, from max-marginals (best_path's scores run
    forward and backward), averaged like the positions. One frame alone misses what the seconds
    around it say. Where every place is near the path there is nothing to tell apart: infinity."""
    T, n = Z.shape
    src = np.maximum(nbr, 0)
    cost = np.where(nbr >= 0, cost, np.inf).astype(np.float32)
    jump = np.float32(jump)
    fwd = np.empty((T, n), np.float32)
    fwd[0] = Z[0]
    for t in range(1, T):
        fwd[t] = Z[t] + np.maximum((fwd[t - 1][src] - cost).max(1), fwd[t - 1].max() - jump)
    bwd = np.zeros((T, n), np.float32)
    for t in range(T - 2, -1, -1):
        nxt = Z[t + 1] + bwd[t + 1]
        bwd[t] = np.maximum((nxt[src] - cost).max(1), nxt.max() - jump)
    total = fwd + bwd  # the best score of a path through each place at each second
    p = pos[path]
    margin = np.empty(T)
    for t in range(T):
        far = (np.hypot(pos[:, 0] - p[t, 0], pos[:, 2] - p[t, 2]) > FAR) | (np.abs(pos[:, 1] - p[t, 1]) > FLOOR)
        margin[t] = total[t, path[t]] - total[t, far].max() if far.any() else np.inf
    return smooth(p, window, margin)


def locate(S: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(path, positions, sureness) for frames whose similarities to the places at `pos` are S."""
    Z = zscores(S.astype(np.float32))
    nbr, dist = neighbours(pos)
    cost = STEP_COST * dist
    path = best_path(Z, nbr, cost, JUMP)
    est = smooth(pos[path], WINDOW)
    sure = sureness(Z, nbr, cost, JUMP, pos, path, WINDOW)
    return path, est, sure


def tier(sure: float) -> str:
    """'sure', 'medium' or 'unsure' for one second's sureness."""
    lo, hi = SURE
    return "sure" if sure >= hi else "medium" if sure >= lo else "unsure"
