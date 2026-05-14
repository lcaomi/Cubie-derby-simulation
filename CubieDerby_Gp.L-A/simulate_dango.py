"""Cubie Derby Monte Carlo Simulation — Group L-A (CLI only, no file output).

Usage:
    python simulate_dango.py --n 1000000 --seed 20260514
"""

import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSITIONS = 32
GREEN = frozenset({2, 10, 15, 22})
RED = frozenset({9, 27})
BLACK = frozenset({5, 19})

NAMES = ["菲比", "陆赫斯", "玲奈", "莫宁", "弗诺诺", "长离"]
N_NORMAL = 6
H_IDX = 6

Z = 2.5758293035489004   # 99% confidence z-score

MORNING_CYCLE = (3, 2, 1)

# Character indices
FBI, LHS, LN, MN, FNN, CL = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def simulate_one_game(board, pos, dist):
    """Run one game.  Reuses pre-allocated *board*, *pos*, *dist*.
    Returns a list of 6 integers: rank[char_idx] (1=best, 6=worst).
    """
    # local bindings for speed
    randint = random.randint
    random_random = random.random
    shuffle = random.shuffle

    # --- reset board ---
    for i in range(32):
        board[i].clear()

    # --- initial state: all normals at position 0, random stack order ---
    order = [0, 1, 2, 3, 4, 5]
    shuffle(order)
    board[0][:] = order   # bottom -> top

    for i in range(6):
        pos[i] = 0
        dist[i] = 0
    pos[H_IDX] = 31
    board[31].append(H_IDX)  # Hakibu alone at 31

    morning_idx = 0
    changli_last = False
    round_num = 0

    while True:
        round_num += 1
        hakibu_active = round_num >= 4

        # ----- build action order -----
        actors = [0, 1, 2, 3, 4, 5]
        if hakibu_active:
            actors.append(H_IDX)
        shuffle(actors)

        # Changli "last action" carried over from previous round
        if changli_last:
            try:
                actors.remove(CL)
            except ValueError:
                pass
            else:
                if random_random() < 0.65:
                    actors.append(CL)   # last
                else:
                    actors.insert(randint(0, len(actors)), CL)
        changli_last = False

        # ----- pre-round checks -----
        p4 = pos[FNN]
        st4 = board[p4]
        funono_bottom = (st4 and st4[0] == FNN)

        # Changli: check for next round
        p5 = pos[CL]
        st5 = board[p5]
        if st5 and st5.index(CL) > 0:   # has characters below
            changli_last = True

        # ----- process each actor -----
        for actor in actors:
            if actor == H_IDX:
                # ================================================
                #  HAKIBU
                # ================================================
                steps = randint(1, 6)
                old_p = pos[H_IDX]
                new_p = (old_p - steps) % 32

                # Remove Hakibu (always at bottom, index 0)
                src = board[old_p]
                src.pop(0)
                # Place at bottom of destination
                dst = board[new_p]
                if dst:
                    dst.insert(0, H_IDX)
                else:
                    dst.append(H_IDX)
                pos[H_IDX] = new_p

                # --- tile effects for Hakibu ---
                # "forward" for Hakibu = toward 0 (position - 1)
                # "backward" for Hakibu = toward 31 (position + 1)
                if new_p in GREEN:
                    _move_hakibu(board, pos, new_p, -1)
                elif new_p in RED:
                    _move_hakibu(board, pos, new_p, 1)
                # Black: Hakibu is excluded from shuffle

            else:
                # ================================================
                #  NORMAL CHARACTER
                # ================================================
                # --- determine steps ---
                if actor == MN:                 # Morning: fixed cycle
                    steps = MORNING_CYCLE[morning_idx]
                    morning_idx = (morning_idx + 1) % 3
                else:
                    steps = randint(1, 3)

                if actor == FBI and random_random() < 0.5:   # Phoebe +1
                    steps += 1

                if actor == FNN and funono_bottom:            # Funono +3
                    steps += 3

                if actor == LN:                               # Lingnai
                    if random_random() < 0.2:
                        continue   # cannot move, skip tile effects
                    if random_random() < 0.6:
                        steps *= 2

                # --- move character + everyone above ---
                old_p = pos[actor]
                src = board[old_p]
                idx = src.index(actor)
                count = len(src) - idx
                # Extract moving group (preserving order)
                moving = src[idx:]
                del src[idx:]

                new_p = (old_p + steps) % 32
                for c in moving:
                    pos[c] = new_p
                    dist[c] += steps

                # Place on top of destination
                board[new_p].extend(moving)

                # --- tile effects ---
                if new_p in GREEN:
                    extra = 1
                    if LHS in moving:           # Luches green bonus
                        extra += 3
                    new_p2 = (new_p + extra) % 32
                    # Remove moving group from current position
                    dst_stack = board[new_p]
                    del dst_stack[-count:]
                    for c in moving:
                        pos[c] = new_p2
                        dist[c] += extra
                    board[new_p2].extend(moving)
                    new_p = new_p2

                elif new_p in RED:
                    penalty = 1
                    if LHS in moving:           # Luches red penalty
                        penalty += 1
                    new_p2 = (new_p - penalty) % 32
                    dst_stack = board[new_p]
                    del dst_stack[-count:]
                    for c in moving:
                        pos[c] = new_p2
                        dist[c] -= penalty
                    board[new_p2].extend(moving)
                    new_p = new_p2

                elif new_p in BLACK:
                    # Shuffle all normal characters at this position
                    bstack = board[new_p]
                    nidxs = [i for i, c in enumerate(bstack) if c < N_NORMAL]
                    if len(nidxs) > 1:
                        chars = [bstack[i] for i in nidxs]
                        shuffle(chars)
                        for i, ch in zip(nidxs, chars):
                            bstack[i] = ch

                # --- end-condition check ---
                for c in moving:
                    if dist[c] >= 31 and pos[c] == 31:
                        return _compute_rankings(board, pos, dist)

        # ----- end-of-round: Hakibu teleport check -----
        if hakibu_active:
            hp = pos[H_IDX]
            for i in range(N_NORMAL):
                if pos[i] > hp:
                    # Teleport Hakibu back to 31
                    board[hp].remove(H_IDX)
                    if board[31]:
                        board[31].insert(0, H_IDX)
                    else:
                        board[31].append(H_IDX)
                    pos[H_IDX] = 31
                    break


def _move_hakibu(board, pos, old_p, delta):
    """Move Hakibu by *delta* (forward direction: -1 toward 0, +1 toward 31)."""
    src = board[old_p]
    src.remove(H_IDX)
    new_p = (old_p + delta) % 32
    dst = board[new_p]
    if dst:
        dst.insert(0, H_IDX)
    else:
        dst.append(H_IDX)
    pos[H_IDX] = new_p


def _compute_rankings(board, pos, dist):
    """Return rank[char_idx] (1-6) for all normal characters."""
    ranks = [0] * N_NORMAL
    rank = 1

    # Characters at position 31: top-to-bottom in stack
    for c in reversed(board[31]):
        if c < N_NORMAL:
            ranks[c] = rank
            rank += 1

    # Remaining characters: sort by (-distance, -position, -stack_index)
    remaining = []
    for c in range(N_NORMAL):
        if ranks[c] == 0:
            p = pos[c]
            st = board[p]
            si = st.index(c) if c in st else 0
            remaining.append((c, dist[c], p, si))

    remaining.sort(key=lambda x: (-x[1], -x[2], -x[3]))

    for c, _, _, _ in remaining:
        ranks[c] = rank
        rank += 1

    return ranks


# ---------------------------------------------------------------------------
# Multiprocessing batch runner
# ---------------------------------------------------------------------------

def _run_batch(args):
    """Run a batch of simulations in a worker process.
    *args* is (batch_size, seed) — must be picklable on Windows.
    """
    batch_size, seed = args
    random.seed(seed)

    # Per-worker allocations
    board = [[] for _ in range(32)]
    pos = [0] * (N_NORMAL + 1)
    dist = [0] * N_NORMAL

    wins = [0] * N_NORMAL
    rank_counts = [[0] * N_NORMAL for _ in range(N_NORMAL)]
    rank_sum = [0.0] * N_NORMAL
    rank_sum_sq = [0.0] * N_NORMAL

    for _ in range(batch_size):
        ranks = simulate_one_game(board, pos, dist)
        for c in range(N_NORMAL):
            r = ranks[c]
            if r == 1:
                wins[c] += 1
            rank_counts[c][r - 1] += 1
            rank_sum[c] += r
            rank_sum_sq[c] += r * r

    return (wins, rank_counts, rank_sum, rank_sum_sq, batch_size)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(wins, n, z=Z):
    """Wilson score confidence interval for a proportion, 99% by default."""
    p_hat = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def normal_ci(mean, std, n, z=Z):
    """Normal-approximation CI for a mean."""
    se = std / math.sqrt(n)
    return mean - z * se, mean + z * se


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def print_results(n, seed, stats):
    """Print the full results table to stdout."""
    print("=" * 72)
    print("团子竞速游戏 Monte Carlo 模拟结果")
    print("=" * 72)
    print()
    print(f"模拟次数：{n:,}")
    print(f"随机种子：{seed}")
    print("置信水平：99%")
    print()
    print("规则假设：")
    print("  1. 特殊格效果不递归触发。")
    print("  2. 哈基布从第 4 回合开始加入行动序列。")
    print("  3. 移动团子组到达已有堆叠时，放在原堆叠上方。")
    print("  4. 玲奈技能先判定 20% 无法移动，再判定 60% 双倍移动。")
    print("  5. 未抵达终点的团子按累计前进距离从高到低补全排名。")
    print()

    # --- Win rate & average rank ---
    print("-" * 72)
    print("胜率与平均名次")
    print("-" * 72)
    header = f"{'团子':<8} {'胜率':>12} {'99%CI':>30} {'平均名次':>12} {'99%CI':>30}"
    print(header)

    for name in NAMES:
        s = stats[name]
        wr = s["wins"] / n
        ci_lo, ci_hi = s["win_ci"]
        avg_r = s["avg_rank"]
        r_lo, r_hi = s["rank_ci"]
        ci_text = f"[{ci_lo:.6f}, {ci_hi:.6f}]"
        rci_text = f"[{r_lo:.4f}, {r_hi:.4f}]"
        print(f"{name:<8} {wr:>12.6f} {ci_text:>30} {avg_r:>12.4f} {rci_text:>30}")

    print()

    # --- Rank distribution ---
    print("-" * 72)
    print("名次概率分布")
    print("-" * 72)
    rank_headers = " ".join(f"{'第'+str(i+1)+'名':>12}" for i in range(6))
    print(f"{'团子':<8} {rank_headers}")

    for name in NAMES:
        s = stats[name]
        probs = " ".join(f"{s['rank_counts'][i]/n:>12.6f}" for i in range(6))
        print(f"{name:<8} {probs}")

    print()

    # --- Conclusions ---
    print("-" * 72)
    print("结论")
    print("-" * 72)

    best_win = max(NAMES, key=lambda nm: stats[nm]["wins"])
    best_avg = min(NAMES, key=lambda nm: stats[nm]["avg_rank"])
    best_stable = min(NAMES, key=lambda nm: stats[nm]["rank_std"])

    print(f"胜率最高：{best_win}  ({stats[best_win]['wins']/n*100:.2f}%)")
    print(f"平均名次最好：{best_avg}  ({stats[best_avg]['avg_rank']:.4f})")
    print(f"表现最稳定：{best_stable}  (名次标准差 {stats[best_stable]['rank_std']:.4f})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="团子竞速 Monte Carlo 模拟 (CLI only)"
    )
    parser.add_argument(
        "--n", type=int, default=1_000_000,
        help="模拟次数 (默认: 1,000,000)"
    )
    parser.add_argument(
        "--seed", type=int, default=20260514,
        help="随机种子 (默认: 20260514)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="并行 worker 进程数 (默认: CPU 核心数)"
    )
    args = parser.parse_args()

    n = args.n
    seed = args.seed
    n_workers = args.workers or os.cpu_count() or 4

    # Ensure UTF-8 output on Windows
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    # --- statistics accumulators ---
    stats = {}
    for name in NAMES:
        stats[name] = {
            "wins": 0,
            "rank_counts": [0] * N_NORMAL,
            "rank_sum": 0.0,
            "rank_sum_sq": 0.0,
        }

    # --- run simulations (multiprocessing) ---
    batch_size = n // n_workers
    remainder = n % n_workers

    work = []
    for i in range(n_workers):
        size = batch_size + (1 if i < remainder else 0)
        if size > 0:
            work.append((size, seed + i))

    sys.stdout.write(
        f"运行 {n:,} 次模拟中...  (使用 {len(work)} 个 worker 进程)\n"
    )
    sys.stdout.flush()

    t0 = time.perf_counter()
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_run_batch, w) for w in work]

        for fut in as_completed(futures):
            wins, rank_counts, rank_sum, rank_sum_sq, batch_n = fut.result()
            completed += batch_n
            pct = completed / n * 100
            elapsed = time.perf_counter() - t0
            eta = elapsed / completed * (n - completed) if completed > 0 else 0
            sys.stdout.write(
                f"  进度: {completed:,}/{n:,} ({pct:.0f}%)  "
                f"已用时: {elapsed:.1f}s  预计剩余: {eta:.1f}s\n"
            )
            sys.stdout.flush()

            # Accumulate
            for c in range(N_NORMAL):
                name = NAMES[c]
                s = stats[name]
                s["wins"] += wins[c]
                s["rank_sum"] += rank_sum[c]
                s["rank_sum_sq"] += rank_sum_sq[c]
                for rk in range(N_NORMAL):
                    s["rank_counts"][rk] += rank_counts[c][rk]

    elapsed = time.perf_counter() - t0
    print(f"\n模拟完成，总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)\n")

    # --- compute derived statistics ---
    for name in NAMES:
        s = stats[name]
        s["avg_rank"] = s["rank_sum"] / n
        var = s["rank_sum_sq"] / n - s["avg_rank"] ** 2
        s["rank_std"] = math.sqrt(var * n / (n - 1)) if n > 1 else 0.0
        s["win_ci"] = wilson_ci(s["wins"], n)
        s["rank_ci"] = normal_ci(s["avg_rank"], s["rank_std"], n)

    # --- print ---
    print_results(n, seed, stats)


if __name__ == "__main__":
    main()
