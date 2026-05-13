"""
团子赛跑 (Cube Derby) Monte Carlo Simulation — Group C-B
=========================================================
High-performance simulation using numba JIT + linked-list board + multiprocessing.

Game: 32-position circular board (0–31), 6 regular characters + 哈基布 (special).
The first character to pass point 0 and arrive at point 31 wins.
Runs ≥1,000,000 simulations; computes 99% Wilson confidence intervals.

Usage:
    python simulate.py [num_simulations] [--workers N] [--test]
"""

import numpy as np
from numba import njit
from collections import defaultdict
from typing import Dict, Tuple
import time
import sys
from multiprocessing import Pool, cpu_count

# =============================================================================
# Constants
# =============================================================================

NUM_POSITIONS: int = 32

DANGO_NAMES = [
    '奥古斯塔',   # 0: Augusta
    '尤诺',       # 1: Yuno
    '弗洛洛',     # 2: Frollo
    '长离',       # 3: Changli
    '今汐',       # 4: Jinxi
    '卡卡罗',     # 5: Kakaro
]

Z_99: float = 2.575829303548901

# =============================================================================
# Inline tile type checks
# =============================================================================

@njit(inline='always', cache=True)
def _is_green(p):
    return p == 2 or p == 10 or p == 15 or p == 22

@njit(inline='always', cache=True)
def _is_red(p):
    return p == 9 or p == 27

@njit(inline='always', cache=True)
def _is_black(p):
    return p == 5 or p == 19

# =============================================================================
# SplitMix64 RNG — state held in 1-element uint64 array
# =============================================================================

@njit(inline='always', cache=True)
def _rng_next(state):
    s = state[0]
    s = np.uint64(s + np.uint64(0x9E3779B97F4A7C15))
    z = np.uint64(s)
    z = np.uint64((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
    z = np.uint64((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
    state[0] = s
    return np.uint32(z ^ (z >> np.uint64(31)))

@njit(inline='always', cache=True)
def _rng_randint(state, lo, hi):
    """Random int in [lo, hi)."""
    return lo + int(_rng_next(state)) % (hi - lo)

@njit(inline='always', cache=True)
def _rng_random(state):
    """Random float in [0, 1)."""
    return float(_rng_next(state)) * (1.0 / 4294967296.0)

@njit(inline='always', cache=True)
def _rng_shuffle(state, arr):
    """Fisher-Yates shuffle in-place."""
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = int(_rng_next(state)) % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]

# =============================================================================
# Linked-list board helpers
#
#   tile_bot[t] = bottom dango on tile t  (-1 = empty)
#   above[d]    = dango directly above d   (-1 = top)
#   below[d]    = dango directly below d   (-1 = bottom)
#   pos[d]      = tile d is on
# =============================================================================

@njit(inline='always', cache=True)
def _tile_append(tile_bot, above, below, pos, t, d):
    """Add dango d to top of tile t."""
    old_bot = tile_bot[t]
    if old_bot == -1:
        tile_bot[t] = d
        below[d] = -1
        above[d] = -1
    else:
        cur = old_bot
        while above[cur] != -1:
            cur = above[cur]
        above[cur] = d
        below[d] = cur
        above[d] = -1
    pos[d] = t

@njit(inline='always', cache=True)
def _tile_remove_one(tile_bot, above, below, pos, t, d):
    """Remove single dango d from its tile (must be on tile t)."""
    d_above = above[d]
    d_below = below[d]
    if d_below != -1:
        above[d_below] = d_above
    else:
        tile_bot[t] = d_above
    if d_above != -1:
        below[d_above] = d_below
    above[d] = -1
    below[d] = -1

@njit(inline='always', cache=True)
def _tile_count(tile_bot, above, t):
    """Return number of dango on tile t."""
    cnt = 0
    cur = tile_bot[t]
    while cur != -1:
        cnt += 1
        cur = above[cur]
    return cnt

@njit(inline='always', cache=True)
def _tile_collect(tile_bot, above, t, out_arr):
    """Collect all dango on tile t into out_arr (bottom-to-top). Returns count."""
    n = 0
    cur = tile_bot[t]
    while cur != -1:
        out_arr[n] = cur
        n += 1
        cur = above[cur]
    return n

@njit(inline='always', cache=True)
def _stack_idx(tile_bot, above, t, d):
    """Return 0-based stack index of d on tile t (0 = bottom)."""
    idx = 0
    cur = tile_bot[t]
    while cur != -1:
        if cur == d:
            return idx
        cur = above[cur]
        idx += 1
    return 0

# =============================================================================
# Move group: d + everything above it from from_t → top of to_t
# =============================================================================

@njit(cache=True)
def _move_group(tile_bot, above, below, pos, from_t, d, to_t):
    """
    Move dango `d` and everything above it on tile `from_t`
    to the top of tile `to_t`. Returns array of moved dango (bottom-to-top).
    """
    moved = np.zeros(7, dtype=np.int32)
    n = 0
    cur = d
    while cur != -1:
        moved[n] = cur
        n += 1
        cur = above[cur]

    d_below = below[d]
    if d_below != -1:
        above[d_below] = -1
    else:
        tile_bot[from_t] = -1
    below[d] = -1

    old_bot = tile_bot[to_t]
    if old_bot == -1:
        tile_bot[to_t] = d
    else:
        cur = old_bot
        while above[cur] != -1:
            cur = above[cur]
        above[cur] = d
        below[d] = cur

    for k in range(n):
        pos[moved[k]] = to_t

    return moved[:n]

# =============================================================================
# Map effects
# =============================================================================

@njit(cache=True)
def _effect_green_red(tile_bot, above, below, pos, fwd, moved, forward):
    """Shift each dango in `moved` by 1 tile (green: forward, red: backward)."""
    for k in range(len(moved)):
        d = moved[k]
        if d == 6:
            continue
        if forward and pos[d] == 31 and fwd[d] > 0:
            continue  # already finished, don't move
        p = pos[d]
        _tile_remove_one(tile_bot, above, below, pos, p, d)
        if forward:
            np_tile = (p + 1) % 32
            fwd[d] += 1
        else:
            np_tile = (p - 1) % 32
        _tile_append(tile_bot, above, below, pos, np_tile, d)

@njit(cache=True)
def _effect_green_red_hakibu(tile_bot, above, below, pos, forward):
    """Shift 哈基布 by 1 tile for green/red effects."""
    d = 6
    p = pos[d]
    _tile_remove_one(tile_bot, above, below, pos, p, d)
    if forward:
        np_tile = (p + 1) % 32
    else:
        np_tile = (p - 1) % 32
    _tile_append(tile_bot, above, below, pos, np_tile, d)

@njit(cache=True)
def _effect_black(state, tile_bot, above, below, pos, t):
    """Randomly re-stack non-hakibu dango on tile t."""
    all_d = np.zeros(7, dtype=np.int32)
    n = _tile_collect(tile_bot, above, t, all_d)
    if n <= 1:
        return

    non_h = np.zeros(n, dtype=np.int32)
    n_nh = 0
    h_present = False
    for i in range(n):
        if all_d[i] == 6:
            h_present = True
        else:
            non_h[n_nh] = all_d[i]
            n_nh += 1

    if n_nh <= 1:
        return

    _rng_shuffle(state, non_h[:n_nh])

    tile_bot[t] = -1
    for i in range(n):
        above[all_d[i]] = -1
        below[all_d[i]] = -1

    if h_present:
        _tile_append(tile_bot, above, below, pos, t, 6)
    for i in range(n_nh):
        _tile_append(tile_bot, above, below, pos, t, non_h[i])

# =============================================================================
# Utility moves
# =============================================================================

@njit(inline='always', cache=True)
def _dango_to_bottom(tile_bot, above, below, pos, t, d):
    """Move dango d to the very bottom of tile t."""
    _tile_remove_one(tile_bot, above, below, pos, t, d)
    old_bot = tile_bot[t]
    if old_bot == -1:
        tile_bot[t] = d
        above[d] = -1
        below[d] = -1
    else:
        below[old_bot] = d
        above[d] = old_bot
        below[d] = -1
        tile_bot[t] = d
    pos[d] = t

@njit(inline='always', cache=True)
def _dango_teleport(tile_bot, above, below, pos, d, to_t):
    """Teleport dango d to tile to_t (placed on top)."""
    old = pos[d]
    _tile_remove_one(tile_bot, above, below, pos, old, d)
    _tile_append(tile_bot, above, below, pos, to_t, d)

# =============================================================================
# Yuno skill: teleport others to Yuno's position
# =============================================================================

@njit(cache=True)
def _trigger_yuno(tile_bot, above, below, pos, fwd, finished):
    """Yuno (1): teleport all other active regular dango to Yuno's tile.
    Preserves original rank order (more forward steps = higher in stack)."""
    target = pos[1]
    others = np.zeros(6, dtype=np.int32)
    n_others = 0
    for j in range(6):
        if j != 1 and finished[j] == 0:
            others[n_others] = j
            n_others += 1
    if n_others == 0:
        return

    for k in range(n_others):
        j = others[k]
        _tile_remove_one(tile_bot, above, below, pos, pos[j], j)

    # Sort by fwd ascending (worse = lower in stack = placed first)
    for i in range(n_others):
        for j2 in range(i + 1, n_others):
            if fwd[others[i]] > fwd[others[j2]]:
                tmp = others[i]
                others[i] = others[j2]
                others[j2] = tmp

    for k in range(n_others):
        _tile_append(tile_bot, above, below, pos, target, others[k])

# =============================================================================
# Win check
# =============================================================================

@njit(inline='always', cache=True)
def _check_win(moved, pos, fwd, finished, rank, finish_counter):
    """
    Check moved group (top-to-bottom) for winners at pos 31.
    All qualifying dango in the stack get ranked by stack order.
    Returns (finish_counter, game_over).
    """
    n = len(moved)
    game_over = False
    for k in range(n - 1, -1, -1):  # top-to-bottom
        d = moved[k]
        if d == 6 or finished[d]:
            continue
        if pos[d] == 31 and fwd[d] > 0:
            finish_counter += 1
            finished[d] = 1
            rank[d] = finish_counter
            game_over = True
    return finish_counter, game_over

# =============================================================================
# Core game simulation (numba JIT nopython)
# =============================================================================

@njit(cache=True)
def simulate_one_game(seed):
    """
    Run one complete game.
    Returns int32 array of length 6: finish_rank[0:6] (1-indexed, 1=first).
    """
    # ---- Board state ----
    tile_bot = np.full(32, -1, dtype=np.int32)
    above = np.full(7, -1, dtype=np.int32)
    below = np.full(7, -1, dtype=np.int32)
    pos = np.zeros(7, dtype=np.int32)
    fwd = np.zeros(7, dtype=np.int32)
    finished = np.zeros(7, dtype=np.int32)
    rank = np.zeros(7, dtype=np.int32)

    # Per-character skill state
    skip = np.zeros(7, dtype=np.int32)
    last_nxt = np.zeros(7, dtype=np.int32)
    changli_last = np.zeros(7, dtype=np.int32)
    forno_bot = np.zeros(7, dtype=np.int32)
    yuno_used = np.zeros(7, dtype=np.int32)
    yuno_mid = np.zeros(7, dtype=np.int32)
    base_dice = np.zeros(7, dtype=np.int32)

    # RNG state
    rng = np.zeros(1, dtype=np.uint64)
    rng[0] = np.uint64(seed)

    # ---- Initial placement: 6 regular dango shuffled at position 0 ----
    order_arr = np.arange(6, dtype=np.int32)
    _rng_shuffle(rng, order_arr)
    for i in range(6):
        _tile_append(tile_bot, above, below, pos, 0, order_arr[i])

    # 哈基布 (index 6) at position 31
    _tile_append(tile_bot, above, below, pos, 31, 6)

    finish_counter = 0
    round_num = 0
    game_over = False

    while not game_over:
        round_num += 1

        # ===== Pre-generate dice =====
        for d in range(6):
            if finished[d] == 0:
                base_dice[d] = _rng_randint(rng, 1, 4)  # 1–3
        if round_num >= 4:
            base_dice[6] = _rng_randint(rng, 1, 7)  # 1–6

        # ===== Pre-round skill checks =====
        for d in range(6):
            if finished[d]:
                continue
            p = pos[d]
            is_top = (above[d] == -1)
            is_bot = (below[d] == -1)

            # 奥古斯塔 (0): if on top → skip this round, move last next round
            if d == 0 and is_top:
                skip[d] = 1
                last_nxt[d] = 1

            # 弗洛洛 (2): if on bottom → +3 extra steps when moving
            forno_bot[d] = 1 if (d == 2 and is_bot) else 0

            # 长离 (3): if has dango below → 65% move last next round
            if d == 3 and below[d] != -1:
                if _rng_random(rng) < 0.65:
                    changli_last[d] = 1

            # 今汐 (4): if has dango above → 40% move to top of stack
            if d == 4 and above[d] != -1:
                if _rng_random(rng) < 0.40:
                    _tile_remove_one(tile_bot, above, below, pos, p, d)
                    _tile_append(tile_bot, above, below, pos, p, d)

            # 尤诺 (1): once per game, after passing midpoint (pos >= 16)
            if d == 1 and yuno_used[d] == 0:
                if pos[d] >= 16:
                    yuno_mid[d] = 1
                if yuno_mid[d]:
                    has_others = False
                    for j in range(6):
                        if j != 1 and finished[j] == 0:
                            has_others = True
                            break
                    if has_others:
                        yuno_used[1] = 1
                        _trigger_yuno(tile_bot, above, below, pos, fwd, finished)

        # ===== Build move order =====
        active_arr = np.zeros(7, dtype=np.int32)
        n_active = 0
        for i in range(6):
            if finished[i] == 0:
                active_arr[n_active] = i
                n_active += 1
        if round_num >= 4:
            active_arr[n_active] = 6
            n_active += 1

        _rng_shuffle(rng, active_arr[:n_active])

        normal = np.zeros(7, dtype=np.int32)
        last = np.zeros(7, dtype=np.int32)
        n_normal = 0
        n_last = 0

        for i in range(n_active):
            d = active_arr[i]
            if skip[d]:
                skip[d] = 0
                continue
            if last_nxt[d] or changli_last[d]:
                last[n_last] = d
                n_last += 1
                last_nxt[d] = 0
                changli_last[d] = 0
            else:
                normal[n_normal] = d
                n_normal += 1

        if n_last > 0:
            _rng_shuffle(rng, last[:n_last])

        order = np.zeros(n_normal + n_last, dtype=np.int32)
        for i in range(n_normal):
            order[i] = normal[i]
        for i in range(n_last):
            order[n_normal + i] = last[i]
        n_order = n_normal + n_last

        # ===== Execute moves =====
        max_reg_pos = -1

        for idx in range(n_order):
            if game_over:
                break

            d = order[idx]
            if finished[d]:
                continue

            p = pos[d]

            # Determine steps
            if d == 6:
                # 哈基布: moves backward (31 → 0 direction)
                total_steps = base_dice[6]
                forward_moving = False
            else:
                dice = base_dice[d]
                extra = 0
                if forno_bot[d]:
                    extra += 3
                    forno_bot[d] = 0
                if d == 5:
                    # 卡卡罗: if in last place → +3
                    min_f = 999999
                    for j in range(6):
                        if finished[j] == 0 and fwd[j] < min_f:
                            min_f = fwd[j]
                    if fwd[d] == min_f:
                        extra += 3
                total_steps = dice + extra
                forward_moving = True

            # Destination
            if forward_moving:
                new_p = (p + total_steps) % 32
            else:
                new_p = (p - total_steps) % 32

            # Move group
            moved = _move_group(tile_bot, above, below, pos, p, d, new_p)

            if forward_moving:
                for k in range(len(moved)):
                    md = moved[k]
                    fwd[md] += total_steps
                    if md != 6 and pos[md] > max_reg_pos:
                        max_reg_pos = pos[md]

                # Win check
                finish_counter, win = _check_win(
                    moved, pos, fwd, finished, rank, finish_counter)
                if win:
                    game_over = True

            if game_over:
                break

            # ---- Tile effects on destination ----
            if _is_green(new_p):
                if forward_moving:
                    _effect_green_red(tile_bot, above, below, pos, fwd,
                                      moved, forward=True)
                    finish_counter, win = _check_win(
                        moved, pos, fwd, finished, rank, finish_counter)
                    if win:
                        game_over = True
                else:
                    _effect_green_red_hakibu(tile_bot, above, below, pos,
                                             forward=True)

            elif _is_red(new_p):
                if forward_moving:
                    _effect_green_red(tile_bot, above, below, pos, fwd,
                                      moved, forward=False)
                else:
                    _effect_green_red_hakibu(tile_bot, above, below, pos,
                                             forward=False)

            elif _is_black(new_p):
                _effect_black(rng, tile_bot, above, below, pos, new_p)

            if game_over:
                break

            # 哈基ブ encounter: if on same tile as active regular dango → bottom
            if d == 6:
                hp = pos[6]
                cur = tile_bot[hp]
                has_active = False
                while cur != -1:
                    if cur != 6 and finished[cur] == 0:
                        has_active = True
                        break
                    cur = above[cur]
                if has_active:
                    _dango_to_bottom(tile_bot, above, below, pos, hp, 6)

        # ===== 哈基ブ teleport: if surpassed → reset to 31 =====
        if not game_over and max_reg_pos > pos[6]:
            _dango_teleport(tile_bot, above, below, pos, 6, 31)

    # ---- Rank remaining (non-finished) dango by position then stack height ----
    remaining = np.zeros(6, dtype=np.int32)
    n_rem = 0
    for d in range(6):
        if finished[d] == 0:
            remaining[n_rem] = d
            n_rem += 1

    # Sort descending by (fwd, stack_idx): higher fwd = better, higher stack = better
    for i in range(n_rem):
        for j in range(i + 1, n_rem):
            di = remaining[i]
            dj = remaining[j]
            hi = _stack_idx(tile_bot, above, pos[di], di)
            hj = _stack_idx(tile_bot, above, pos[dj], dj)
            if fwd[di] < fwd[dj] or (fwd[di] == fwd[dj] and hi < hj):
                remaining[i], remaining[j] = remaining[j], remaining[i]

    for i in range(n_rem):
        rank[remaining[i]] = finish_counter + 1 + i

    result = np.zeros(6, dtype=np.int32)
    for i in range(6):
        result[i] = rank[i]
    return result


# =============================================================================
# Batch runner (multiprocessing worker)
# =============================================================================

def run_batch(args: Tuple[int, int]) -> Dict[int, Dict[int, int]]:
    """
    Run `n_sims` simulations starting from seed `batch_id * 1_000_000`.
    Returns {dango_idx: {rank: count}}.
    """
    batch_id, n_sims = args
    base = batch_id * 1_000_000
    results = {i: defaultdict(int) for i in range(6)}

    for k in range(n_sims):
        ranks = simulate_one_game(base + k)
        for d_idx in range(6):
            results[d_idx][ranks[d_idx]] += 1

    return results


def merge(results: Dict[int, Dict[int, int]],
          batch: Dict[int, Dict[int, int]]):
    """Merge batch results into the global results dict."""
    for d in range(6):
        for rank_val, cnt in batch[d].items():
            results[d][rank_val] += cnt


# =============================================================================
# Statistics
# =============================================================================

def wilson_ci(successes: int, total: int) -> Tuple[float, float, float]:
    """Wilson score interval (99%). Returns (p_hat, lower, upper)."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z2 = Z_99 ** 2
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = Z_99 * np.sqrt(
        (p * (1.0 - p) + z2 / (4.0 * total)) / total
    ) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def fmt_dur(sec: float) -> str:
    """Format duration in human-readable form."""
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    h, m = divmod(int(sec), 3600)
    return f"{h}h{m // 60}m"


# =============================================================================
# Verbose test mode (pure-Python reference for debugging)
# =============================================================================

class VerboseGame:
    """Pure-Python reference implementation for debugging one game."""

    def __init__(self, seed: int):
        import random
        self.rng = random.Random(seed)
        self.board = [[] for _ in range(NUM_POSITIONS)]
        self.pos = [0] * 7
        self.fwd = [0] * 7
        self.finished = [False] * 7
        self.rank = [0] * 7
        self.finish_counter = 0
        self.round_num = 0
        self.game_over = False
        self.skip = [False] * 7
        self.move_last = [False] * 7
        self.changli_last = [False] * 7
        self.forno_bot = [False] * 7
        self.yuno_used = [False] * 7
        self.yuno_mid = [False] * 7
        self.base_dice = [0] * 7

        order = list(range(6))
        self.rng.shuffle(order)
        for d in order:
            self.board[0].append(d)
        self.pos[6] = 31
        self.board[31].append(6)

    def _find(self, d):
        p = self.pos[d]
        try:
            idx = self.board[p].index(d)
            return p, idx
        except ValueError:
            for pp in range(NUM_POSITIONS):
                if d in self.board[pp]:
                    self.pos[d] = pp
                    return pp, self.board[pp].index(d)
        raise RuntimeError(f"Dango {d} missing")

    def _move_group(self, from_p, stack_idx, steps, forward):
        tile = self.board[from_p]
        moved = tile[stack_idx:]
        del tile[stack_idx:]
        new_p = (from_p + steps) % 32 if forward else (from_p - steps) % 32
        for d in moved:
            self.pos[d] = new_p
            if forward and d != 6:
                self.fwd[d] += steps
        self.board[new_p].extend(moved)
        return new_p, moved

    def _check_win(self, moved):
        for d in reversed(moved):
            if d == 6 or self.finished[d]:
                continue
            if self.pos[d] == 31 and self.fwd[d] > 0:
                self.finished[d] = True
                self.finish_counter += 1
                self.rank[d] = self.finish_counter
                self.game_over = True

    def _green_red(self, moved, forward):
        for d in moved:
            if d == 6:
                continue
            if forward and self.pos[d] == 31 and self.fwd[d] > 0:
                continue
            p = self.pos[d]
            self.board[p].remove(d)
            np_tile = (p + 1) % 32 if forward else (p - 1) % 32
            self.pos[d] = np_tile
            self.board[np_tile].append(d)
            if forward:
                self.fwd[d] += 1

    def _black(self, p):
        tile = self.board[p]
        non_h = [d for d in tile if d != 6]
        h_list = [d for d in tile if d == 6]
        if len(non_h) > 1:
            self.rng.shuffle(non_h)
        self.board[p] = h_list + non_h

    def _yuno(self):
        self.yuno_used[1] = True
        target = self.pos[1]
        others = [j for j in range(6) if j != 1 and not self.finished[j]]
        if not others:
            return
        for j in others:
            self.board[self.pos[j]].remove(j)
        others.sort(key=lambda j: self.fwd[j])
        for j in others:
            self.pos[j] = target
            self.board[target].append(j)

    def run(self):
        while not self.game_over:
            self.round_num += 1
            print(f"\n{'='*50}")
            print(f"Round {self.round_num}")
            print(f"{'='*50}")

            # Pre-generate dice
            for d in range(6):
                if not self.finished[d]:
                    self.base_dice[d] = self.rng.randint(1, 3)
            if self.round_num >= 4:
                self.base_dice[6] = self.rng.randint(1, 6)

            # Pre-round skills
            for d in range(6):
                if self.finished[d]:
                    continue
                p, sidx = self._find(d)
                tile = self.board[p]
                is_top = (sidx == len(tile) - 1)
                is_bot = (sidx == 0)

                if d == 0 and is_top:
                    self.skip[d] = True
                    self.move_last[d] = True
                self.forno_bot[d] = (d == 2 and is_bot)
                if d == 3 and sidx > 0:
                    if self.rng.random() < 0.65:
                        self.changli_last[d] = True
                if d == 4 and sidx < len(tile) - 1:
                    if self.rng.random() < 0.40:
                        tile.remove(d)
                        tile.append(d)
                if d == 1 and not self.yuno_used[d]:
                    if self.pos[d] >= 16:
                        self.yuno_mid[d] = True
                    if self.yuno_mid[d]:
                        if any(not self.finished[j] for j in range(6) if j != 1):
                            self._yuno()

            # Move order
            active = [i for i in range(6) if not self.finished[i]]
            if self.round_num >= 4:
                active.append(6)
            self.rng.shuffle(active)

            normal, last_g = [], []
            for d in active:
                if self.skip[d]:
                    self.skip[d] = False
                    continue
                if self.move_last[d] or self.changli_last[d]:
                    last_g.append(d)
                    self.move_last[d] = False
                    self.changli_last[d] = False
                else:
                    normal.append(d)
            self.rng.shuffle(last_g)
            order = normal + last_g

            names = [DANGO_NAMES[d] if d < 6 else '哈基布' for d in order]
            print(f"Move order: {names}")

            max_reg_pos = -1
            for d in order:
                if self.game_over or self.finished[d]:
                    continue
                name = DANGO_NAMES[d] if d < 6 else '哈基布'
                p, sidx = self._find(d)
                dice = self.base_dice[d]

                if d == 6:
                    new_p, moved = self._move_group(p, sidx, dice, False)
                    print(f"  哈基布 at pos={p} rolls {dice} -> pos={new_p}")
                    if new_p in (2, 10, 15, 22):
                        self._move_group(new_p, self.board[new_p].index(6), 1, True)
                        print(f"    Green! -> pos={self.pos[6]}")
                    elif new_p in (9, 27):
                        self._move_group(new_p, self.board[new_p].index(6), 1, False)
                        print(f"    Red! -> pos={self.pos[6]}")
                    elif new_p in (5, 19):
                        self._black(new_p)
                        print(f"    Black! re-stacked")
                    # Encounter
                    hp = self.pos[6]
                    has_active = any(d2 != 6 and not self.finished[d2] for d2 in self.board[hp])
                    if has_active:
                        self.board[hp].remove(6)
                        self.board[hp].insert(0, 6)
                        print(f"    Encounter! hakibu to bottom")
                else:
                    extra = 0
                    if self.forno_bot[d]:
                        extra += 3
                        self.forno_bot[d] = False
                    if d == 5:
                        min_f = min(self.fwd[j] for j in range(6) if not self.finished[j])
                        if self.fwd[d] == min_f:
                            extra += 3
                    steps = dice + extra
                    new_p, moved = self._move_group(p, sidx, steps, True)
                    print(f"  {name} at pos={p} rolls {dice}{f'+{extra}' if extra else ''} -> pos={new_p}")
                    for m in moved:
                        if m != 6 and self.pos[m] > max_reg_pos:
                            max_reg_pos = self.pos[m]
                    self._check_win(moved)
                    if self.game_over:
                        print(f"    >>> WIN! Game Over <<<")
                    if new_p in (2, 10, 15, 22):
                        self._green_red(moved, True)
                        print(f"    Green! -> pos={[self.pos[m] for m in moved]}")
                        self._check_win(moved)
                    elif new_p in (9, 27):
                        self._green_red(moved, False)
                        print(f"    Red! -> pos={[self.pos[m] for m in moved]}")
                    elif new_p in (5, 19):
                        self._black(new_p)
                        print(f"    Black! re-stacked")
                if self.game_over:
                    break

            if not self.game_over and max_reg_pos > self.pos[6]:
                print(f"  哈基布 teleports to 31 (max_reg_pos={max_reg_pos} > hakibu={self.pos[6]})")
                self.board[self.pos[6]].remove(6)
                self.pos[6] = 31
                self.board[31].append(6)

            print(f"\n  Board after round {self.round_num}:")
            for pp in range(NUM_POSITIONS):
                if self.board[pp]:
                    tile_names = [DANGO_NAMES[d2] if d2 < 6 else '哈基ブ' for d2 in self.board[pp]]
                    print(f"    pos {pp}: {tile_names}")

        # Rank remaining
        remaining = [d for d in range(6) if not self.finished[d]]
        remaining.sort(key=lambda d: (
            self.fwd[d],
            self._find(d)[1]
        ), reverse=True)
        for i, d in enumerate(remaining):
            self.rank[d] = self.finish_counter + 1 + i
        return self.rank[:6]


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Cube Derby Monte Carlo Simulation — Group C-B (numba-optimized)')
    parser.add_argument('sims', nargs='?', type=int, default=1_000_000,
                        help='Number of simulations (default: 1,000,000)')
    parser.add_argument('--workers', '-w', type=int, default=max(1, cpu_count() - 1),
                        help=f'Worker processes (default: cpu_count-1 = {max(1, cpu_count()-1)})')
    parser.add_argument('--batch', '-b', type=int, default=50000,
                        help='Batch size per worker task (default: 50000)')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run a single verbose game for debugging')
    parser.add_argument('--test-n', type=int, default=1,
                        help='Number of verbose test games to run')
    args = parser.parse_args()

    # --- Test mode ---
    if args.test:
        print("=" * 50)
        print("  VERBOSE TEST MODE — Group C-B")
        print("=" * 50)
        for i in range(args.test_n):
            seed = i * 1000
            print(f"\n\n>>> Game {i+1} (seed={seed}) <<<")
            g = VerboseGame(seed)
            ranks = g.run()
            print(f"\n  Final ranking:")
            for d in sorted(range(6), key=lambda x: ranks[x]):
                print(f"    Rank {ranks[d]}: {DANGO_NAMES[d]}")
        return

    # --- Production run ---
    TOTAL = args.sims
    WORKERS = args.workers
    BATCH = args.batch

    print("=" * 70)
    print("  Cube Derby — Group C-B Monte Carlo Simulation")
    print("  Engine: numba JIT + linked-list board + multiprocessing")
    print("=" * 70)
    print(f"  Simulations: {TOTAL:,}")
    print(f"  Workers:     {WORKERS}")
    print(f"  Batch size:  {BATCH:,}")
    print(f"  Confidence:  99% (Wilson score interval)")
    print("=" * 70)
    print()

    # Warm up: force numba JIT compilation
    print("  Compiling simulation engine (numba JIT)...", end=" ", flush=True)
    _ = simulate_one_game(0)
    print("done.\n")

    # Build batch list
    full_batches = TOTAL // BATCH
    remainder = TOTAL % BATCH
    batches = [(i, BATCH) for i in range(full_batches)]
    if remainder:
        batches.append((full_batches, remainder))

    print(f"  Total batches: {len(batches)}")
    print()

    results = {i: defaultdict(int) for i in range(6)}
    completed = 0
    t0 = time.time()

    try:
        with Pool(processes=WORKERS) as pool:
            chunk = max(1, WORKERS * 2)
            for start in range(0, len(batches), chunk):
                group = batches[start:start + chunk]
                for batch_res in pool.map(run_batch, group):
                    merge(results, batch_res)
                    completed += BATCH

                elapsed = time.time() - t0
                done = min((start + len(group)) * BATCH, TOTAL)
                rate = done / elapsed if elapsed > 0 else 0
                eta = (TOTAL - done) / rate if rate > 0 else 0

                print(f"  [{fmt_dur(elapsed)}] "
                      f"{done:,}/{TOTAL:,} "
                      f"({100*done/TOTAL:.1f}%)  "
                      f"{rate:,.0f} sim/s  "
                      f"ETA {fmt_dur(eta)}")

    except KeyboardInterrupt:
        print("\n  Interrupted. Computing results from partial data...")
        TOTAL = max(1, completed)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  RESULTS  ({TOTAL:,} simulations, {fmt_dur(elapsed)})")
    print(f"{'='*70}")

    # --- Win probability with 99% CI ---
    header = f"  {'Character':<12} {'Win %':>10}  {'99% CI':>22}  {'Wins':>10}"
    print(header)
    print(f"  {'-'*12} {'-'*10}  {'-'*22}  {'-'*10}")

    for d in range(6):
        wins = results[d].get(1, 0)
        p, lo, hi = wilson_ci(wins, TOTAL)
        ci_str = f"[{100*lo:.4f}%, {100*hi:.4f}%]"
        print(f"  {DANGO_NAMES[d]:<12} {100*p:>9.4f}%  {ci_str:>22}  {wins:>10,}")

    # --- Rank distribution ---
    print(f"\n  Rank Distribution:")
    print(f"  {'Character':<12}", end="")
    for r in range(1, 7):
        print(f"  Rank{r}   ", end="")
    print(f"\n  {'-'*12}", end="")
    for _ in range(6):
        print(f"  ------", end="")
    print()
    for d in range(6):
        print(f"  {DANGO_NAMES[d]:<12}", end="")
        for r in range(1, 7):
            cnt = results[d].get(r, 0)
            print(f"  {100*cnt/TOTAL:5.2f}%", end="")
        print()

    # --- Podium (Top-3) ---
    print(f"\n  Podium (Top-3) Probability:")
    print(f"  {'Character':<12} {'Top-3 %':>10}  {'99% CI':>22}")
    print(f"  {'-'*12} {'-'*10}  {'-'*22}")
    for d in range(6):
        podium = sum(results[d].get(r, 0) for r in (1, 2, 3))
        p, lo, hi = wilson_ci(podium, TOTAL)
        ci_str = f"[{100*lo:.4f}%, {100*hi:.4f}%]"
        print(f"  {DANGO_NAMES[d]:<12} {100*p:>9.4f}%  {ci_str:>22}")

    # --- Validation ---
    total_win = sum(results[d].get(1, 0) for d in range(6))
    print(f"\n  Sum of win rates: {100*total_win/TOTAL:.4f}%  "
          f"{'(should be ~100%)' if abs(total_win/TOTAL - 1.0) < 0.002 else 'WARNING: check logic!'}")

    # --- Average rank ---
    print(f"\n  Average Rank:")
    for d in range(6):
        avg = sum(r * cnt for r, cnt in results[d].items()) / TOTAL
        print(f"  {DANGO_NAMES[d]:<12} {avg:.4f}")

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
