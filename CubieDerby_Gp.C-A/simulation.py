"""
Board game simulation - Monte Carlo win probability for 团子 (Dango) racing game.
Optimized with numba JIT + linked-list board representation.

Game: 32-position circular board (0-31), 6 regular characters + 哈基ブ (special).
Runs millions of simulations and computes 99% Wilson confidence intervals.

Usage:
    python simulation.py [num_simulations] [--workers N] [--test]
"""

import numpy as np
from numba import njit
from collections import defaultdict
from typing import List, Tuple, Dict
import time
import sys
from multiprocessing import Pool, cpu_count

# =============================================================================
# Constants
# =============================================================================

NUM_POSITIONS: int = 32

# Tile effect positions (sets for verbose/debug mode)
_GREEN = {2, 10, 15, 22}
_RED = {9, 27}
_BLACK = {5, 19}

# Numba-compatible tile effect checks (inline helpers)
@njit(inline='always', cache=True)
def _is_green(p):
    return p == 2 or p == 10 or p == 15 or p == 22

@njit(inline='always', cache=True)
def _is_red(p):
    return p == 9 or p == 27

@njit(inline='always', cache=True)
def _is_black(p):
    return p == 5 or p == 19

DANGO_NAMES: List[str] = [
    '奥古斯塔',   # 0: Augusta
    '尤诺',       # 1: Yuno
    '弗诺诺',     # 2: Fornono
    '长离',       # 3: Changli
    '今汐',       # 4: Jinxi
    '卡卡罗',     # 5: Kakaro
]

Z_99: float = 2.575829303548901

# =============================================================================
# Fast inline RNG (SplitMix64)  — state held in 1-element uint64 array
# =============================================================================

@njit(inline='always', cache=True)
def _rng_next(state):
    """Return random uint32. state[0] is updated in-place."""
    s = state[0]
    s = np.uint64(s + np.uint64(0x9E3779B97F4A7C15))
    z = np.uint64(s)
    z = np.uint64((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
    z = np.uint64((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
    state[0] = s
    return np.uint32(z ^ (z >> np.uint64(31)))


@njit(inline='always', cache=True)
def _rng_randint(state, lo, hi):
    """Return random int in [lo, hi). state[0] is updated."""
    return lo + int(_rng_next(state)) % (hi - lo)


@njit(inline='always', cache=True)
def _rng_random(state):
    """Return random float in [0, 1). state[0] is updated."""
    return float(_rng_next(state)) * (1.0 / 4294967296.0)


@njit(inline='always', cache=True)
def _rng_shuffle(state, arr):
    """Fisher-Yates shuffle in-place. state[0] is updated."""
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = int(_rng_next(state)) % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]


# =============================================================================
# Numba-compatible board helpers (linked-list representation)
#
#  tile_bot[t] = bottom dango on tile t  (-1 = empty)
#  above[d]    = dango directly above d   (-1 = top)
#  below[d]    = dango directly below d   (-1 = bottom)
#  pos[d]      = tile d is on
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
        # Walk to current top
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
    # pos[d] intentionally not cleared (caller will update)


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
def _tile_has_active_forward(tile_bot, above, finished, t):
    """True if tile t has any non-finished, non-hakibu dango."""
    cur = tile_bot[t]
    while cur != -1:
        if cur != 6 and finished[cur] == 0:
            return True
        cur = above[cur]
    return False


@njit(cache=True)
def _move_group(tile_bot, above, below, pos, from_t, d, to_t):
    """
    Move dango `d` and everything above it on tile `from_t`
    to the top of tile `to_t`.
    Returns array of moved dango (bottom-to-top).
    """
    # Collect moved dango (bottom-to-top, starting from d)
    moved = np.zeros(7, dtype=np.int32)
    n = 0
    cur = d
    while cur != -1:
        moved[n] = cur
        n += 1
        cur = above[cur]

    # Cut from source tile
    d_below = below[d]
    if d_below != -1:
        above[d_below] = -1
    else:
        tile_bot[from_t] = -1
    below[d] = -1

    # Attach to destination tile
    old_bot = tile_bot[to_t]
    if old_bot == -1:
        tile_bot[to_t] = d
    else:
        # Walk to top of destination
        cur = old_bot
        while above[cur] != -1:
            cur = above[cur]
        above[cur] = d
        below[d] = cur

    # Update positions
    for k in range(n):
        pos[moved[k]] = to_t

    return moved[:n]


@njit(inline='always', cache=True)
def _check_wins(moved, pos, fwd, finished, rank, finish_counter, target_total):
    """
    Check for winners among moved dango. Top-to-bottom = best rank on tie.
    Returns (finish_counter, game_over).
    """
    n = len(moved)
    for k in range(n - 1, -1, -1):  # top-to-bottom
        d = moved[k]
        if d == 6 or finished[d]:
            continue
        if pos[d] == 31 and fwd[d] > 0:
            finish_counter += 1
            finished[d] = 1
            rank[d] = finish_counter
            if finish_counter >= target_total:
                return finish_counter, True
    return finish_counter, False


@njit(cache=True)
def _effect_green_red(tile_bot, above, below, pos, fwd, moved, forward):
    """Move each dango in `moved` by 1 tile. Modifies pos arrays."""
    for k in range(len(moved)):
        d = moved[k]
        if d == 6 or (forward and pos[d] == 31 and fwd[d] > 0):
            # Don't move finished dango or 哈基布
            continue
        p = pos[d]
        _tile_remove_one(tile_bot, above, below, pos, p, d)
        if forward:
            np_tile = (p + 1) % 32
            fwd[d] += 1
        else:
            np_tile = (p - 1) % 32
        _tile_append(tile_bot, above, below, pos, np_tile, d)


@njit(cache=True)
def _effect_black(state, tile_bot, above, below, pos, t):
    """Randomly re-stack non-hakibu dango on tile t."""
    # Collect all dango on this tile
    all_d = np.zeros(7, dtype=np.int32)
    n = _tile_collect(tile_bot, above, t, all_d)

    if n <= 1:
        return

    # Separate hakibu from non-hakibu
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

    # Shuffle non-hakibu
    _rng_shuffle(state, non_h[:n_nh])

    # Clear tile but preserve hakibu's presence
    tile_bot[t] = -1
    for i in range(n):
        above[all_d[i]] = -1
        below[all_d[i]] = -1

    # Rebuild: hakibu at bottom, then shuffled non-hakibu
    if h_present:
        _tile_append(tile_bot, above, below, pos, t, 6)
    for i in range(n_nh):
        _tile_append(tile_bot, above, below, pos, t, non_h[i])


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


@njit(cache=True)
def _trigger_yuno(tile_bot, above, below, pos, fwd, finished):
    """Yuno skill: teleport all other active regular dango to Yuno's tile."""
    target = pos[1]

    # Collect others
    others = np.zeros(6, dtype=np.int32)
    n_others = 0
    for j in range(6):
        if j != 1 and finished[j] == 0:
            others[n_others] = j
            n_others += 1

    if n_others == 0:
        return

    # Remove from current tiles
    for k in range(n_others):
        j = others[k]
        _tile_remove_one(tile_bot, above, below, pos, pos[j], j)

    # Sort by forward_steps (ascending = worse rank = lower in stack)
    for i in range(n_others):
        for j2 in range(i + 1, n_others):
            if fwd[others[i]] > fwd[others[j2]]:
                tmp = others[i]
                others[i] = others[j2]
                others[j2] = tmp

    # Place on target tile (last placed = highest = best rank)
    for k in range(n_others):
        _tile_append(tile_bot, above, below, pos, target, others[k])


# =============================================================================
# Core game simulation  (numba JIT nopython)
# =============================================================================

@njit(cache=True)
def simulate_one_game(seed):
    """
    Run one game to completion.
    Returns int32 array of length 6: finish_rank[0:6] (1-indexed, 1=first).
    """
    # ---- All state as local arrays (numba allocates on stack) ----
    tile_bot = np.full(32, -1, dtype=np.int32)
    above = np.full(7, -1, dtype=np.int32)
    below = np.full(7, -1, dtype=np.int32)
    pos = np.zeros(7, dtype=np.int32)
    fwd = np.zeros(7, dtype=np.int32)
    finished = np.zeros(7, dtype=np.int32)
    rank = np.zeros(7, dtype=np.int32)

    skip = np.zeros(7, dtype=np.int32)
    last_nxt = np.zeros(7, dtype=np.int32)
    changli_last = np.zeros(7, dtype=np.int32)
    forno_bot = np.zeros(7, dtype=np.int32)
    yuno_used = np.zeros(7, dtype=np.int32)
    yuno_mid = np.zeros(7, dtype=np.int32)

    # RNG state as 1-element array (passed by reference to inlined helpers)
    rng = np.zeros(1, dtype=np.uint64)
    rng[0] = np.uint64(seed)

    # ---- Initial placement ----
    # 6 regular dango shuffled at position 0
    order_arr = np.arange(6, dtype=np.int32)
    _rng_shuffle(rng, order_arr)
    for i in range(6):
        _tile_append(tile_bot, above, below, pos, 0, order_arr[i])

    # 哈基ブ (6) at position 31
    _tile_append(tile_bot, above, below, pos, 31, 6)

    finish_counter = 0
    round_num = 0
    game_over = False

    while not game_over:
        round_num += 1

        # ===== PRE-ROUND SKILLS =====
        for d in range(6):
            if finished[d]:
                continue

            p = pos[d]
            is_top = (above[d] == -1)
            is_bot = (below[d] == -1)

            # 奥古斯塔 (0): top -> skip round, move last next round
            if d == 0 and is_top:
                skip[d] = 1
                last_nxt[d] = 1

            # 弗诺诺 (2): bottom at round start -> +3 steps when moving
            forno_bot[d] = 1 if (d == 2 and is_bot) else 0

            # 长离 (3): has dango below -> 65% move last next round
            if d == 3 and below[d] != -1:
                if _rng_random(rng) < 0.65:
                    changli_last[d] = 1

            # 今汐 (4): has dango above -> 40% move to top of this tile
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

        # ===== MOVE ORDER =====
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

        # Final move order
        order = np.zeros(n_normal + n_last, dtype=np.int32)
        for i in range(n_normal):
            order[i] = normal[i]
        for i in range(n_last):
            order[n_normal + i] = last[i]
        n_order = n_normal + n_last

        # ===== EXECUTE MOVES =====
        max_fwd = -1

        for idx in range(n_order):
            if game_over:
                break

            d = order[idx]
            if finished[d]:
                continue

            p = pos[d]

            # Dice roll + extra steps
            if d == 6:
                dice = _rng_randint(rng, 1, 7)  # 1-6
                total_steps = dice
                forward_moving = False
            else:
                dice = _rng_randint(rng, 1, 4)  # 1-3
                extra = 0
                if forno_bot[d]:
                    extra += 3
                    forno_bot[d] = 0
                if d == 5:
                    # Kakaro: last place check
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
                # Update forward steps
                for k in range(len(moved)):
                    md = moved[k]
                    fwd[md] += total_steps
                    if md != 6 and pos[md] > max_fwd:
                        max_fwd = pos[md]

                # Win check (top-to-bottom in moved)
                finish_counter, game_over = _check_wins(
                    moved, pos, fwd, finished, rank, finish_counter, 6)

            if game_over:
                break

            # ---- Tile effects on new position ----
            if _is_green(new_p):
                if forward_moving:
                    _effect_green_red(tile_bot, above, below, pos, fwd,
                                      moved, forward=True)
                    finish_counter, game_over = _check_wins(
                        moved, pos, fwd, finished, rank, finish_counter, 6)
                else:
                    _effect_green_red(tile_bot, above, below, pos, fwd,
                                      np.array([6], dtype=np.int32), forward=True)
            elif _is_red(new_p):
                if forward_moving:
                    _effect_green_red(tile_bot, above, below, pos, fwd,
                                      moved, forward=False)
                else:
                    _effect_green_red(tile_bot, above, below, pos, fwd,
                                      np.array([6], dtype=np.int32), forward=False)
            elif _is_black(new_p):
                _effect_black(rng, tile_bot, above, below, pos, new_p)

            if game_over:
                break

            # 哈基ブ encounter: if landed on tile with active forward dango
            if d == 6:
                hp = pos[6]
                if _tile_has_active_forward(tile_bot, above, finished, hp):
                    _dango_to_bottom(tile_bot, above, below, pos, hp, 6)

        # ===== 哈基ブ TELEPORT CHECK =====
        if not game_over and max_fwd > pos[6]:
            _dango_teleport(tile_bot, above, below, pos, 6, 31)

    # Return finish ranks for 6 regular dango
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
    for d in range(6):
        for rank, cnt in batch[d].items():
            results[d][rank] += cnt


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


# =============================================================================
# Formatting
# =============================================================================

def fmt_dur(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    h, m = divmod(int(sec), 3600)
    return f"{h}h{m // 60}m"


# =============================================================================
# Verbose test mode (pure Python reference implementation for debugging)
# =============================================================================

class VerboseGame:
    """Pure-Python reference implementation for debugging."""

    def __init__(self, seed: int):
        import random
        self.rng = random.Random(seed)
        self.board = [[] for _ in range(NUM_POSITIONS)]
        self.positions = [0] * 7
        self.forward_steps = [0] * 7
        self.finished = [False] * 7
        self.finish_rank = [0] * 7
        self.finish_counter = 0
        self.round_num = 0
        self.game_over = False
        self.skip_round = [False] * 7
        self.move_last_next = [False] * 7
        self.changli_move_last = [False] * 7
        self.fornono_bottom = [False] * 7
        self.yuno_used = [False] * 7
        self.yuno_past_midpoint = [False] * 7

        # Initial stacking
        order = list(range(6))
        self.rng.shuffle(order)
        for d_idx in order:
            self.board[0].append(d_idx)
            self.positions[d_idx] = 0
        self.positions[6] = 31
        self.board[31].append(6)

    def _find(self, d_idx):
        pos = self.positions[d_idx]
        tile = self.board[pos]
        for i, d in enumerate(tile):
            if d == d_idx:
                return pos, i
        for p in range(NUM_POSITIONS):
            for i, d in enumerate(self.board[p]):
                if d == d_idx:
                    self.positions[d_idx] = p
                    return p, i
        raise RuntimeError(f"Dango {d_idx} missing")

    def _is_last_place(self, d_idx):
        best = 10**9
        for i in range(6):
            if not self.finished[i] and self.forward_steps[i] < best:
                best = self.forward_steps[i]
        return self.forward_steps[d_idx] == best

    def _move_group(self, from_pos, stack_idx, steps, forward):
        tile = self.board[from_pos]
        moved = tile[stack_idx:]
        del tile[stack_idx:]
        new_pos = (from_pos + steps) % NUM_POSITIONS if forward else (from_pos - steps) % NUM_POSITIONS
        for d in moved:
            self.positions[d] = new_pos
            if forward:
                self.forward_steps[d] += steps
        self.board[new_pos].extend(moved)
        return new_pos, moved

    def _move_specific(self, from_pos, dango_list, steps, forward):
        tile = self.board[from_pos]
        for d in dango_list:
            tile.remove(d)
        new_pos = (from_pos + steps) % NUM_POSITIONS if forward else (from_pos - steps) % NUM_POSITIONS
        for d in dango_list:
            self.positions[d] = new_pos
            if forward:
                self.forward_steps[d] += steps
        self.board[new_pos].extend(dango_list)

    def _check_wins(self, dango_group):
        for d in reversed(dango_group):
            if d == 6 or self.finished[d]:
                continue
            if self.positions[d] == 31 and self.forward_steps[d] > 0:
                self.finished[d] = True
                self.finish_counter += 1
                self.finish_rank[d] = self.finish_counter
                if self.finish_counter >= 6:
                    self.game_over = True

    def _trigger_yuno(self):
        self.yuno_used[1] = True
        target = self.positions[1]
        others = [j for j in range(6) if j != 1 and not self.finished[j]]
        if not others:
            return
        for j in others:
            self.board[self.positions[j]].remove(j)
        others.sort(key=lambda j: self.forward_steps[j])
        for j in others:
            self.positions[j] = target
            self.board[target].append(j)

    def _get_move_extra(self, d_idx):
        extra = 0
        if self.fornono_bottom[d_idx]:
            extra += 3
            self.fornono_bottom[d_idx] = False
        if d_idx == 5 and self._is_last_place(d_idx):
            extra += 3
        return extra

    def _apply_pre_round_skills(self):
        for d in range(6):
            if self.finished[d]:
                continue
            pos, sidx = self._find(d)
            tile = self.board[pos]
            tile_len = len(tile)
            is_top = (sidx == tile_len - 1)
            is_bot = (sidx == 0)

            if d == 0 and is_top:
                self.skip_round[d] = True
                self.move_last_next[d] = True
            self.fornono_bottom[d] = (d == 2 and is_bot)
            if d == 3 and sidx > 0:
                if self.rng.random() < 0.65:
                    self.changli_move_last[d] = True
            if d == 4 and sidx < tile_len - 1:
                if self.rng.random() < 0.40:
                    tile.remove(d)
                    tile.append(d)
            if d == 1 and not self.yuno_used[d]:
                if self.positions[d] >= 16:
                    self.yuno_past_midpoint[d] = True
                if self.yuno_past_midpoint[d]:
                    if any(not self.finished[j] for j in range(6) if j != 1):
                        self._trigger_yuno()

    def _get_move_order(self):
        active = [i for i in range(6) if not self.finished[i]]
        if self.round_num >= 4:
            active.append(6)
        self.rng.shuffle(active)
        normal, last = [], []
        for d in active:
            if self.skip_round[d]:
                self.skip_round[d] = False
                continue
            if self.move_last_next[d] or self.changli_move_last[d]:
                last.append(d)
                self.move_last_next[d] = False
                self.changli_move_last[d] = False
            else:
                normal.append(d)
        self.rng.shuffle(last)
        return normal + last

    def execute_round(self):
        self.round_num += 1
        print(f"\n{'='*50}")
        print(f"Round {self.round_num}")
        print(f"{'='*50}")
        self._apply_pre_round_skills()
        order = self._get_move_order()
        names = [DANGO_NAMES[i] if i < 6 else '哈基布' for i in order]
        print(f"Move order: {names}")

        max_fwd = -1
        for d in order:
            if self.game_over or self.finished[d]:
                continue
            name = DANGO_NAMES[d] if d < 6 else '哈基布'
            pos, sidx = self._find(d)
            dice = self.rng.randint(1, 6) if d == 6 else self.rng.randint(1, 3)
            print(f"\n  {name} at pos={pos} stack_idx={sidx} rolls {dice}")

            if d == 6:
                new_pos, moved = self._move_group(pos, sidx, dice, False)
                print(f"    哈基布 moves backward {dice}: pos {pos} -> {new_pos}")
                if new_pos in _GREEN:
                    self._move_specific(new_pos, moved, 1, True)
                    print(f"    Green! -> pos {self.positions[6]}")
                elif new_pos in _RED:
                    self._move_specific(new_pos, moved, 1, False)
                    print(f"    Red! -> pos {self.positions[6]}")
                elif new_pos in _BLACK:
                    tile = self.board[new_pos]
                    non_h = [d2 for d2 in tile if d2 != 6]
                    h_list = [d2 for d2 in tile if d2 == 6]
                    if len(non_h) > 1:
                        self.rng.shuffle(non_h)
                    self.board[new_pos] = h_list + non_h
                    print(f"    Black! re-stacked")
                # Encounter check
                has_active = any(d2 != 6 and not self.finished[d2] for d2 in self.board[self.positions[6]])
                if has_active:
                    tile = self.board[self.positions[6]]
                    tile.remove(6)
                    tile.insert(0, 6)
                    print(f"    Encounter! 哈基布 to bottom")
            else:
                extra = self._get_move_extra(d)
                if extra:
                    print(f"    Extra steps: +{extra}")
                new_pos, moved = self._move_group(pos, sidx, dice + extra, True)
                print(f"    Moves forward {dice+extra}: pos {pos} -> {new_pos}")
                for m in moved:
                    if m != 6 and self.positions[m] > max_fwd:
                        max_fwd = self.positions[m]
                self._check_wins(moved)
                if self.game_over:
                    print(f"    >>> GAME OVER <<<")
                if new_pos in _GREEN:
                    self._move_specific(new_pos, moved, 1, True)
                    print(f"    Green! -> pos {[self.positions[m] for m in moved]}")
                    self._check_wins(moved)
                elif new_pos in _RED:
                    self._move_specific(new_pos, moved, 1, False)
                    print(f"    Red! -> pos {[self.positions[m] for m in moved]}")
                elif new_pos in _BLACK:
                    tile = self.board[new_pos]
                    non_h = [d2 for d2 in tile if d2 != 6]
                    h_list = [d2 for d2 in tile if d2 == 6]
                    if len(non_h) > 1:
                        self.rng.shuffle(non_h)
                    self.board[new_pos] = h_list + non_h
                    print(f"    Black! re-stacked at pos {new_pos}")
            if self.game_over:
                break

        if not self.game_over and max_fwd > self.positions[6]:
            print(f"\n  哈基布 teleports to 31 (max_fwd={max_fwd} > hakibu={self.positions[6]})")
            self.board[self.positions[6]].remove(6)
            self.positions[6] = 31
            self.board[31].append(6)

        print(f"\n  Board state after round {self.round_num}:")
        for p in range(NUM_POSITIONS):
            if self.board[p]:
                tile_names = [(DANGO_NAMES[d2] if d2 < 6 else '哈基ブ') for d2 in self.board[p]]
                print(f"    pos {p}: {tile_names}")
        if not self.game_over:
            print(f"  Forward steps: {dict(zip(DANGO_NAMES, self.forward_steps[:6]))}")

    def run(self):
        while not self.game_over:
            self.execute_round()
        return self.finish_rank[:6]


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Dango Racing Game Monte Carlo Simulation (numba-optimized)')
    parser.add_argument('sims', nargs='?', type=int, default=10_000_000,
                        help='Number of simulations (default: 10,000,000)')
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
        print("  VERBOSE TEST MODE")
        print("=" * 50)
        for i in range(args.test_n):
            seed = i * 1000
            print(f"\n\n>>> Game {i+1} (seed={seed}) <<<")
            g = VerboseGame(seed)
            ranks = g.run()
            print(f"\n  Finish order:")
            for d_idx in sorted(range(6), key=lambda i: ranks[i]):
                print(f"    Rank {ranks[d_idx]}: {DANGO_NAMES[d_idx]}")
        return

    # --- Production run ---
    TOTAL = args.sims
    WORKERS = args.workers
    BATCH = args.batch

    print("=" * 70)
    print("  团子 (Dango) Racing Game - Monte Carlo Simulation")
    print("  Engine: numba JIT + linked-list board")
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

    print(f"  Batches: {len(batches)}")
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
        print("\n  Interrupted. Computing from partial data...")
        TOTAL = completed

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  RESULTS  ({TOTAL:,} simulations, {fmt_dur(elapsed)})")
    print(f"{'='*70}")

    # --- Win probabilities with 99% CI ---
    header = f"  {'Character':<12} {'Win %':>10}  {'99% CI':>20}  {' Wins '}"
    print(header)
    print(f"  {'-'*12} {'-'*10}  {'-'*20}  {'-'*8}")

    for d in range(6):
        wins = results[d].get(1, 0)
        p, lo, hi = wilson_ci(wins, TOTAL)
        ci_str = f"[{100*lo:.4f}%, {100*hi:.4f}%]"
        print(f"  {DANGO_NAMES[d]:<12} {100*p:>9.4f}%  {ci_str:>20}  {wins:>8,}")

    # --- Full rank distribution ---
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

    # --- Top-3 (podium) probability ---
    print(f"\n  Podium (Top-3) Probability:")
    print(f"  {'Character':<12} {'Top-3 %':>10}  {'99% CI':>20}")
    print(f"  {'-'*12} {'-'*10}  {'-'*20}")
    for d in range(6):
        podium = sum(results[d].get(r, 0) for r in (1, 2, 3))
        p, lo, hi = wilson_ci(podium, TOTAL)
        ci_str = f"[{100*lo:.4f}%, {100*hi:.4f}%]"
        print(f"  {DANGO_NAMES[d]:<12} {100*p:>9.4f}%  {ci_str:>20}")

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
