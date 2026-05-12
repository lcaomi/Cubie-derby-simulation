"""
团子快跑 (Dango Race) Monte Carlo Simulation — Group B
======================================================
Optimized with numba JIT + linked-list board + multiprocessing.

Game: 32-position circular board (0-31), 6 regular characters + 哈基布 (special).
Game ends when ANY regular character reaches position 31.
Runs millions of simulations and computes 99% Wilson confidence intervals.

Usage:
    python simulate.py --trials 1000000 --seed 42 --output results.csv
    python simulate.py --trials 1000 --seed 0          # quick test

Rule assumptions:
  1. 哈基布 starts acting from round 4, AFTER all regular characters
  2. Green/Red tile effects do NOT chain-trigger
  3. 爱弥斯 skill: if no one ahead when passing midpoint, skill stays available
  4. 千咲 compares BASE dice (pre-skill), not final move distance
  5. 莫宁 cycle advances on each actual move
  6. Game ends immediately when ANY regular character reaches position 31
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

DANGO_NAMES: List[str] = [
    '千咲',       # 0: Chisaki
    '莫宁',       # 1: Moning
    '琳奈',       # 2: Linne
    '爱弥斯',     # 3: Amis
    '守岸人',     # 4: Shouganjin
    '柯莱塔',     # 5: Keleita
]

Z_99: float = 2.575829303548901

# =============================================================================
# Fast inline RNG (SplitMix64) — state held in 1-element uint64 array
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
    return lo + int(_rng_next(state)) % (hi - lo)


@njit(inline='always', cache=True)
def _rng_random(state):
    return float(_rng_next(state)) * (1.0 / 4294967296.0)


@njit(inline='always', cache=True)
def _rng_shuffle(state, arr):
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = int(_rng_next(state)) % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]


# =============================================================================
# Numba-compatible board helpers (linked-list representation)
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


@njit(cache=True)
def _move_group(tile_bot, above, below, pos, from_t, d, to_t):
    """
    Move dango `d` and everything above it on tile `from_t`
    to the top of tile `to_t`.
    Returns array of moved dango (bottom-to-top) and count in moved[n].
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
def _is_green(p):
    return p == 2 or p == 10 or p == 15 or p == 22


@njit(inline='always', cache=True)
def _is_red(p):
    return p == 9 or p == 27


@njit(inline='always', cache=True)
def _is_black(p):
    return p == 5 or p == 19


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
# Settlement: assign ranks when game ends (first arrival at 31)
# =============================================================================

@njit
def _settle_ranking(tile_bot, above, pos, arrived, n_arrived, ranks):
    """
    Assign finish ranks when `arrived` group (bottom→top) reaches position 31.
    Top of arrived group gets rank 1, then remaining characters ranked by
    (pos descending, then stack height from bottom descending).
    """
    rank = 1

    # Step 1: Rank the arrived group — top→bottom = best→worst
    for k in range(n_arrived - 1, -1, -1):
        d = arrived[k]
        if d != 6:
            ranks[d] = rank
            rank += 1

    # Step 2: Collect remaining unranked regular characters
    # Sort key: -(pos * 10 + stack_height) for descending priority
    remaining_keys = np.zeros(6, dtype=np.int32)
    remaining_d = np.zeros(6, dtype=np.int32)
    n_rem = 0

    for d in range(6):
        if ranks[d] == 0:
            p = pos[d]
            # Compute stack height from bottom
            h = 0
            cur = tile_bot[p]
            while cur != d and cur != -1:
                h += 1
                cur = above[cur]
            key = -(p * 10 + h)  # negative for ascending sort → descending priority
            remaining_keys[n_rem] = key
            remaining_d[n_rem] = d
            n_rem += 1

    # Bubble sort by key ascending (= priority descending), n_rem ≤ 6
    for i in range(n_rem):
        for j in range(i + 1, n_rem):
            if remaining_keys[i] > remaining_keys[j]:
                tmp_k = remaining_keys[i]
                remaining_keys[i] = remaining_keys[j]
                remaining_keys[j] = tmp_k
                tmp_d = remaining_d[i]
                remaining_d[i] = remaining_d[j]
                remaining_d[j] = tmp_d

    for i in range(n_rem):
        ranks[remaining_d[i]] = rank
        rank += 1


# =============================================================================
# Core game simulation (numba JIT nopython)
# =============================================================================

@njit(cache=True)
def simulate_one_game(seed):
    """
    Run one complete B组 game.
    Returns int32 array of length 6: finish_rank[0:6] (1-indexed, 1=first).
    """
    # ---- Board state ----
    tile_bot = np.full(32, -1, dtype=np.int32)
    above = np.full(7, -1, dtype=np.int32)
    below = np.full(7, -1, dtype=np.int32)
    pos = np.zeros(7, dtype=np.int32)

    # Per-character state
    finished = np.zeros(7, dtype=np.int32)
    ranks = np.zeros(7, dtype=np.int32)

    # RNG state
    rng = np.zeros(1, dtype=np.uint64)
    rng[0] = np.uint64(seed)

    # ---- Initial placement ----
    order_arr = np.arange(6, dtype=np.int32)
    _rng_shuffle(rng, order_arr)
    for i in range(6):
        _tile_append(tile_bot, above, below, pos, 0, order_arr[i])

    # 哈基布 (6) at position 31
    _tile_append(tile_bot, above, below, pos, 31, 6)

    # ---- B组-specific persistent state ----
    moning_cycle = 0          # 0→3, 1→2, 2→1
    amis_skill_used = 0       # boolean

    round_num = 0
    game_over = False

    while not game_over:
        round_num += 1

        # ===== Step 1: Pre-generate base dice =====
        active_arr = np.zeros(6, dtype=np.int32)
        n_active = 0
        for d in range(6):
            if finished[d] == 0:
                active_arr[n_active] = d
                n_active += 1

        if n_active == 0:
            break

        base_dice = np.zeros(7, dtype=np.int32)
        for i in range(n_active):
            d = active_arr[i]
            if d == 1:  # 莫宁: fixed cycle
                base_dice[d] = [3, 2, 1][moning_cycle]
            elif d == 4:  # 守岸人: 2 or 3
                base_dice[d] = 2 if _rng_random(rng) < 0.5 else 3
            else:  # 千咲, 琳奈, 爱弥斯, 柯莱塔: 1~3
                base_dice[d] = _rng_randint(rng, 1, 4)

        # Step 2: Find minimum base dice
        min_dice = 999
        for i in range(n_active):
            if base_dice[active_arr[i]] < min_dice:
                min_dice = base_dice[active_arr[i]]

        # Step 3: Shuffle action order
        _rng_shuffle(rng, active_arr[:n_active])

        # Track max regular position this round (for 哈基布 teleport)
        max_reg_pos = -1

        # ===== Step 4: Execute regular character moves =====
        for idx in range(n_active):
            if game_over:
                break

            d = active_arr[idx]
            bd = base_dice[d]

            # ---- Calculate final steps ----
            if d == 0:  # 千咲: +2 if base dice == min
                steps = bd + (2 if bd == min_dice else 0)
            elif d == 1:  # 莫宁
                steps = bd
                moning_cycle = (moning_cycle + 1) % 3
            elif d == 2:  # 琳奈: 60% double / 20% skip / 20% normal
                r = _rng_random(rng)
                if r < 0.60:
                    steps = bd * 2
                elif r < 0.80:
                    steps = 0
                else:
                    steps = bd
            elif d == 3:  # 爱弥斯: normal
                steps = bd
            elif d == 4:  # 守岸人
                steps = bd
            elif d == 5:  # 柯莱塔: 28% double
                steps = bd * 2 if _rng_random(rng) < 0.28 else bd
            else:
                steps = bd

            # ---- Skip if 0 steps ----
            if steps == 0:
                if pos[d] > max_reg_pos:
                    max_reg_pos = pos[d]
                continue

            # ---- Move group ----
            p = pos[d]
            new_p = p + steps
            if new_p > 31:
                new_p = 31

            moved = _move_group(tile_bot, above, below, pos, p, d, new_p)
            moved_len = len(moved)

            # Update max position
            for k in range(moved_len):
                md = moved[k]
                if md != 6 and pos[md] > max_reg_pos:
                    max_reg_pos = pos[md]

            # ---- Tile effects ----
            eff_p = new_p

            if _is_green(eff_p):
                # Move entire group forward 1
                leader = moved[0]
                to_p = eff_p + 1
                if to_p > 31:
                    to_p = 31
                moved = _move_group(tile_bot, above, below, pos, eff_p, leader, to_p)
                moved_len = len(moved)
                eff_p = to_p
                for k in range(moved_len):
                    md = moved[k]
                    if md != 6 and pos[md] > max_reg_pos:
                        max_reg_pos = pos[md]

            elif _is_red(eff_p):
                # Move entire group backward 1
                leader = moved[0]
                to_p = eff_p - 1
                if to_p < 0:
                    to_p = 0
                moved = _move_group(tile_bot, above, below, pos, eff_p, leader, to_p)
                moved_len = len(moved)
                eff_p = to_p

            elif _is_black(eff_p):
                _effect_black(rng, tile_bot, above, below, pos, eff_p)

            # ---- Game over check: anyone reached 31 ----
            if eff_p == 31:
                _settle_ranking(tile_bot, above, pos, moved, moved_len, ranks)
                game_over = True
                break

            # ---- 爱弥斯 skill check ----
            if d == 3 and amis_skill_used == 0 and pos[3] >= 16:
                best_target = -1
                best_dist = 999
                for c in range(6):
                    if c == 3 or finished[c]:
                        continue
                    cp = pos[c]
                    if cp > pos[3]:
                        dist = cp - pos[3]
                        if dist < best_dist:
                            best_dist = dist
                            best_target = c
                if best_target != -1:
                    amis_skill_used = 1
                    old_p = pos[3]
                    _tile_remove_one(tile_bot, above, below, pos, old_p, 3)
                    target_p = pos[best_target]
                    _tile_append(tile_bot, above, below, pos, target_p, 3)

                    if pos[3] > max_reg_pos:
                        max_reg_pos = pos[3]

        # ===== Step 6: 哈基布 acts (from round 4) =====
        if not game_over and round_num >= 4:
            hp = pos[6]
            h_dice = _rng_randint(rng, 1, 7)  # 1~6

            h_new_p = hp - h_dice
            if h_new_p < 0:
                h_new_p = 0

            _ = _move_group(tile_bot, above, below, pos, hp, 6, h_new_p)
            h_eff_p = h_new_p

            # 哈基布 tile effects: "forward" = toward 0, "backward" = toward 31
            if _is_green(h_eff_p):
                h_to = h_eff_p - 1
                if h_to < 0:
                    h_to = 0
                _move_group(tile_bot, above, below, pos, h_eff_p, 6, h_to)
                h_eff_p = h_to

            elif _is_red(h_eff_p):
                h_to = h_eff_p + 1
                if h_to > 31:
                    h_to = 31
                _move_group(tile_bot, above, below, pos, h_eff_p, 6, h_to)
                h_eff_p = h_to

            elif _is_black(h_eff_p):
                _effect_black(rng, tile_bot, above, below, pos, h_eff_p)

            # 哈基布 encounter: if tile has regular characters → 哈基布 to bottom
            if h_eff_p != 31:  # already at bottom if at 31 initially
                has_active = False
                cur = tile_bot[h_eff_p]
                while cur != -1:
                    if cur != 6:
                        has_active = True
                        break
                    cur = above[cur]
                if has_active:
                    _dango_to_bottom(tile_bot, above, below, pos, h_eff_p, 6)

        # ===== Step 7: 哈基布 teleport check =====
        if not game_over:
            if max_reg_pos > pos[6]:
                _dango_teleport(tile_bot, above, below, pos, 6, 31)

    # Return finish ranks for 6 regular dango
    result = np.zeros(6, dtype=np.int32)
    for i in range(6):
        result[i] = ranks[i]
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
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Dango Racing Game Monte Carlo Simulation — Group B (numba-optimized)')
    parser.add_argument('--trials', '-t', type=int, default=10_000_000,
                        help='Number of simulations (default: 10,000,000)')
    parser.add_argument('--workers', '-w', type=int, default=max(1, cpu_count() - 1),
                        help=f'Worker processes (default: cpu_count-1 = {max(1, cpu_count()-1)})')
    parser.add_argument('--batch', '-b', type=int, default=50000,
                        help='Batch size per worker task (default: 50000)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    TOTAL = args.trials
    WORKERS = args.workers
    BATCH = args.batch
    SEED = args.seed

    print("=" * 70)
    print("  团子 (Dango) Racing Game — Group B Monte Carlo Simulation")
    print("  Engine: numba JIT + linked-list board + multiprocessing")
    print("=" * 70)
    print(f"  Simulations: {TOTAL:,}")
    print(f"  Workers:     {WORKERS}")
    print(f"  Batch size:  {BATCH:,}")
    print(f"  Seed:        {SEED}")
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

    # Offset seeds by master seed
    seed_offset = SEED * 1_000_000
    batches = [(seed_offset + i, n) for i, n in batches]

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
        TOTAL = max(1, completed)

    elapsed = time.time() - t0

    # --- Build stats dict sorted by win rate descending ---
    stats = []
    for d in range(6):
        counter = results[d]
        wins = counter.get(1, 0)
        p_win, ci_lo, ci_hi = wilson_ci(wins, TOTAL)
        rank_sum = sum(r * cnt for r, cnt in counter.items())
        avg_rank = rank_sum / TOTAL if TOTAL > 0 else 0.0
        rank_probs = [counter.get(r, 0) / TOTAL for r in range(1, 7)]
        podium = sum(counter.get(r, 0) for r in (1, 2, 3))
        p_podium, plo, phi = wilson_ci(podium, TOTAL)
        stats.append({
            "name": DANGO_NAMES[d],
            "wins": wins,
            "win_rate": p_win,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "avg_rank": avg_rank,
            "rank_probs": rank_probs,
            "podium_rate": p_podium,
            "podium_lo": plo,
            "podium_hi": phi,
        })

    stats.sort(key=lambda s: s["win_rate"], reverse=True)

    print(f"\n{'='*70}")
    print(f"  RESULTS  ({TOTAL:,} simulations, {fmt_dur(elapsed)})")
    print(f"{'='*70}")

    # --- Win probabilities with 99% CI (sorted) ---
    print(f"\n  Win Probability (sorted):")
    print(f"  {'Rank':<6} {'Character':<12} {'Win %':>10}  {'99% CI':>20}  {'Wins':>10}  {'Avg Rank':>9}")
    print(f"  {'-'*6} {'-'*12} {'-'*10}  {'-'*20}  {'-'*10}  {'-'*9}")
    for i, s in enumerate(stats):
        ci_str = f"[{100*s['ci_lo']:.4f}%, {100*s['ci_hi']:.4f}%]"
        print(f"  {i+1:<6} {s['name']:<12} {100*s['win_rate']:>9.4f}%  {ci_str:>20}  {s['wins']:>10,}  {s['avg_rank']:>8.4f}")

    # --- Full rank distribution (sorted) ---
    print(f"\n  Rank Distribution (sorted by win rate):")
    print(f"  {'Character':<12}", end="")
    for r in range(1, 7):
        print(f"  Rank{r}   ", end="")
    print(f"\n  {'-'*12}", end="")
    for _ in range(6):
        print(f"  ------", end="")
    print()
    for s in stats:
        print(f"  {s['name']:<12}", end="")
        for rp in s["rank_probs"]:
            print(f"  {100*rp:5.2f}%", end="")
        print()

    # --- Podium (Top-3) probability (sorted) ---
    print(f"\n  Podium (Top-3) Probability (sorted):")
    print(f"  {'Character':<12} {'Top-3 %':>10}  {'99% CI':>20}")
    print(f"  {'-'*12} {'-'*10}  {'-'*20}")
    for s in stats:
        ci_str = f"[{100*s['podium_lo']:.4f}%, {100*s['podium_hi']:.4f}%]"
        print(f"  {s['name']:<12} {100*s['podium_rate']:>9.4f}%  {ci_str:>20}")

    # --- Validation ---
    total_win_rate = sum(s["win_rate"] for s in stats)
    print(f"\n  Sum of win rates: {100*total_win_rate:.4f}%  "
          f"{'(should be ~100%)' if abs(total_win_rate - 1.0) < 0.001 else 'WARNING!'}")

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
