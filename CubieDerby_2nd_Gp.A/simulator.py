"""
团子赛跑 Monte Carlo 模拟 — Group A (第二版)
===============================================
基于 numba JIT + 链表棋盘 + 多进程的高性能模拟。

游戏规则：32 个点位（0–31）的环形地图，6 个普通团子 + 哈基布（特殊团子）。
当任意普通团子 linear distance >= 31 时游戏结算。
至少模拟 1,000,000 场；使用 99% Wilson 置信区间。

用法：
    python simulator.py [模拟次数] [--workers N] [--test]

对歧义规则的解释：
  1. "抵达31号点" 实现为 linear forward distance >= 31（允许越过 31 号点）。
  2. 特殊格效果不连锁触发（每次落点只处理一次）。
  3. 哈基布从第 4 回合起加入行动序列。
  4. 绿色/红色格效果作用于该格上全部团子（已完成比赛的除外）。
  5. 奥古斯塔"最后行动"指整个行动序列的最后（包括哈基布之后）。
  6. 尤诺传送排名相邻的团子；原排名较高的在堆叠中靠上。
  7. 卡提希娅"+2格"加在基础骰子结果之上。
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

DUMPLING_NAMES = [
    '奥古斯塔',   # 0
    '尤诺',       # 1
    '今汐',       # 2
    '卡卡罗',     # 3
    '绯雪',       # 4
    '卡提希娅',   # 5
]

Z_99: float = 2.575829303548901

# =============================================================================
# Inline tile type checks
# =============================================================================

@njit(inline='always', cache=True)
def _is_green(p):
    return p == 3 or p == 9 or p == 19


@njit(inline='always', cache=True)
def _is_red(p):
    return p == 15 or p == 25 or p == 29


@njit(inline='always', cache=True)
def _is_black(p):
    return p == 5 or p == 13 or p == 22


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
# 链表棋盘辅助函数
#
#   tile_bot[t] = 格 t 上最底部的团子  (-1 = 空格)
#   above[d]    = 团子 d 正上方的团子   (-1 = 顶部 / 无)
#   below[d]    = 团子 d 正下方的团子   (-1 = 底部 / 无)
#   pos[d]      = 团子 d 所在的格 (0–31)
#   fwd[d]      = 普通团子累计前进距离
# =============================================================================

@njit(inline='always', cache=True)
def _tile_append(tile_bot, above, below, pos, t, d):
    """将团子 d 放到格 t 的堆叠最上方。"""
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
    """将团子 d 从其所在格 t 单独移除。"""
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
def _stack_idx(tile_bot, above, t, d):
    """返回团子 d 在格 t 上的堆叠索引（0 = 底部）。"""
    idx = 0
    cur = tile_bot[t]
    while cur != -1:
        if cur == d:
            return idx
        cur = above[cur]
        idx += 1
    return 0


# =============================================================================
# 移动组：团子 d 及其上方全部团子，从 from_t 移动到 to_t 的堆叠最上方。
# 返回被移动的团子数组（从底部到顶部）。
# =============================================================================

@njit(cache=True)
def _move_group(tile_bot, above, below, pos, from_t, d, to_t):
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
# 特殊格效果
# =============================================================================

@njit(cache=True)
def _effect_green_red(tile_bot, above, below, pos, fwd, finished, moved, forward):
    """
    将 moved 中每个非哈基布、未完成比赛的团子移动 1 格。
    forward=True: 向 31 号方向（pos+1, fwd+1）。
    forward=False: 向 0 号方向（pos-1, fwd-1）。
    """
    for k in range(len(moved)):
        d = moved[k]
        if d == 6:
            continue
        if finished[d]:
            continue
        p = pos[d]
        _tile_remove_one(tile_bot, above, below, pos, p, d)
        if forward:
            np_tile = (p + 1) % 32
            fwd[d] += 1
        else:
            np_tile = (p - 1) % 32
            fwd[d] -= 1
        _tile_append(tile_bot, above, below, pos, np_tile, d)


@njit(cache=True)
def _effect_green_red_hakibu(tile_bot, above, below, pos, forward):
    """对哈基布施加绿色/红色格效果：移动 1 格。"""
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
    """随机重新堆叠格 t 上的普通团子（不含哈基布）。哈基布保持在最底部。"""
    all_d = np.zeros(7, dtype=np.int32)
    n = 0
    cur = tile_bot[t]
    while cur != -1:
        all_d[n] = cur
        n += 1
        cur = above[cur]

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
# 辅助移动函数
# =============================================================================

@njit(inline='always', cache=True)
def _dumpling_to_bottom(tile_bot, above, below, pos, t, d):
    """将团子 d 移动到格 t 的最底部。"""
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
def _dumpling_teleport(tile_bot, above, below, pos, d, to_t):
    """将团子 d 传送至格 to_t（放在堆叠最上方）。"""
    old = pos[d]
    _tile_remove_one(tile_bot, above, below, pos, old, d)
    _tile_append(tile_bot, above, below, pos, to_t, d)


# =============================================================================
# 尤诺技能：将排名相邻的普通团子传送至尤诺所在格
# =============================================================================

@njit(cache=True)
def _trigger_yuno(tile_bot, above, below, pos, fwd, finished):
    """
    尤诺（1）：将当前排名紧邻尤诺前一位和后一位的普通团子（活跃中）
    传送至尤诺所在格。原排名较高的团子在堆叠中靠上。
    """
    target = pos[1]

    # 收集所有活跃的普通团子
    active = np.zeros(6, dtype=np.int32)
    n_active = 0
    for j in range(6):
        if finished[j] == 0:
            active[n_active] = j
            n_active += 1

    if n_active <= 1:
        return

    # 按 (fwd 降序, 堆叠索引 降序) 排序 — fwd 越大/堆叠越靠上 = 名次越好
    for i in range(n_active):
        for j2 in range(i + 1, n_active):
            di = active[i]
            dj = active[j2]
            si = _stack_idx(tile_bot, above, pos[di], di)
            sj = _stack_idx(tile_bot, above, pos[dj], dj)
            if fwd[di] < fwd[dj] or (fwd[di] == fwd[dj] and si < sj):
                active[i], active[j2] = active[j2], active[i]

    # 找到尤诺的排名索引
    yuno_rank = -1
    for i in range(n_active):
        if active[i] == 1:
            yuno_rank = i
            break

    if yuno_rank == -1:
        return

    # 收集排名相邻的团子（前一名、后一名）
    tele = np.zeros(2, dtype=np.int32)
    n_tele = 0
    if yuno_rank > 0:
        tele[n_tele] = active[yuno_rank - 1]
        n_tele += 1
    if yuno_rank < n_active - 1:
        tele[n_tele] = active[yuno_rank + 1]
        n_tele += 1

    if n_tele == 0:
        return

    # 将传送目标团子从原位置移除
    for k in range(n_tele):
        d = tele[k]
        _tile_remove_one(tile_bot, above, below, pos, pos[d], d)

    # 放到尤诺所在格。先放排名较差的，再放排名较好的，使较好者靠上。
    # 按排名从低到高反向遍历。
    for k in range(n_tele - 1, -1, -1):
        _tile_append(tile_bot, above, below, pos, target, tele[k])


# =============================================================================
# 终点检测：检查是否有普通团子满足 fwd >= 31
# =============================================================================

@njit(inline='always', cache=True)
def _check_win(moved, fwd, finished, rank, finish_counter):
    """
    从上到下检查移动组中是否有团子完成比赛。
    完成条件：普通团子的 fwd >= 31。
    返回 (finish_counter, game_over)。
    """
    n = len(moved)
    game_over = False
    for k in range(n - 1, -1, -1):  # 从上到下
        d = moved[k]
        if d == 6 or finished[d]:
            continue
        if fwd[d] >= 31:
            finish_counter += 1
            finished[d] = 1
            rank[d] = finish_counter
            game_over = True
    return finish_counter, game_over


# =============================================================================
# 核心游戏模拟（numba JIT nopython）
# =============================================================================

@njit(cache=True)
def simulate_one_game(seed):
    """
    运行一局完整比赛。
    返回长度为 6 的 int32 数组：finish_rank[0:6]（1 起始，1=第一名）。
    """
    # ---- 棋盘状态 ----
    tile_bot = np.full(32, -1, dtype=np.int32)
    above = np.full(7, -1, dtype=np.int32)
    below = np.full(7, -1, dtype=np.int32)
    pos = np.zeros(7, dtype=np.int32)
    fwd = np.zeros(7, dtype=np.int32)
    finished = np.zeros(7, dtype=np.int32)
    rank = np.zeros(7, dtype=np.int32)

    # 各团子技能状态
    augusta_skip = np.int32(0)       # 奥古斯塔本回合跳过
    augusta_force_last = np.int32(0) # 奥古斯塔下回合最后行动
    yuno_used = np.int32(0)          # 尤诺技能已使用
    yuno_mid = np.int32(0)           # 尤诺已过赛程中点
    feixue_met = np.int32(0)         # 绯雪已遇见哈基布
    cartethyia_unlocked = np.int32(0) # 卡提希娅技能已解锁
    cartethyia_used_this_move = np.int32(0) # 卡提希娅本回合触发标记

    base_dice = np.zeros(7, dtype=np.int32)

    # 随机数状态
    rng = np.zeros(1, dtype=np.uint64)
    rng[0] = np.uint64(seed)

    # ---- 初始放置：6 个普通团子在 0 号位随机堆叠 ----
    order_arr = np.arange(6, dtype=np.int32)
    _rng_shuffle(rng, order_arr)
    for i in range(6):
        _tile_append(tile_bot, above, below, pos, 0, order_arr[i])

    # 哈基布（索引 6）位于 31 号位
    _tile_append(tile_bot, above, below, pos, 31, 6)

    finish_counter = 0
    round_num = 0
    game_over = False

    while not game_over:
        round_num += 1

        # ===== 预生成骰子 =====
        for d in range(6):
            if finished[d] == 0:
                base_dice[d] = _rng_randint(rng, 1, 4)  # 1–3
        if round_num >= 4:
            base_dice[6] = _rng_randint(rng, 1, 7)  # 1–6

        # ===== 回合开始前技能判定 =====
        for d in range(6):
            if finished[d]:
                continue
            p = pos[d]
            is_top = (above[d] == -1)

            # 奥古斯塔 (0)：若在堆叠最顶端 → 本回合不行动，下回合最后行动
            if d == 0 and is_top:
                augusta_skip = 1
                augusta_force_last = 1

            # 今汐 (2)：若头顶有其他团子 → 40% 概率移到堆叠最上方
            if d == 2 and above[d] != -1:
                if _rng_random(rng) < 0.40:
                    _tile_remove_one(tile_bot, above, below, pos, p, d)
                    _tile_append(tile_bot, above, below, pos, p, d)

            # 尤诺 (1)：每场一次，过赛程中点（fwd >= 16）后触发
            if d == 1 and yuno_used == 0:
                if fwd[d] >= 16:
                    yuno_mid = 1
                if yuno_mid:
                    # 检查是否有其他活跃普通团子
                    has_others = False
                    for j in range(6):
                        if j != 1 and finished[j] == 0:
                            has_others = True
                            break
                    if has_others:
                        yuno_used = 1
                        _trigger_yuno(tile_bot, above, below, pos, fwd, finished)

        # ===== 构建行动顺序 =====
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
        last_g = np.zeros(7, dtype=np.int32)
        n_normal = 0
        n_last = 0

        for i in range(n_active):
            d = active_arr[i]
            if d == 0 and augusta_skip:
                augusta_skip = 0
                continue
            if augusta_force_last and d == 0:
                last_g[n_last] = d
                n_last += 1
                augusta_force_last = 0
            else:
                normal[n_normal] = d
                n_normal += 1

        if n_last > 0:
            _rng_shuffle(rng, last_g[:n_last])

        order = np.zeros(n_normal + n_last, dtype=np.int32)
        for i in range(n_normal):
            order[i] = normal[i]
        for i in range(n_last):
            order[n_normal + i] = last_g[i]
        n_order = n_normal + n_last

        # ===== 按顺序依次行动 =====
        max_reg_pos = -1

        for idx in range(n_order):
            if game_over:
                break

            d = order[idx]
            if finished[d]:
                continue

            p = pos[d]

            # --- 决定移动步数 ---
            if d == 6:
                # 哈基布：反向移动（31 → 0 方向）
                total_steps = base_dice[6]
                forward_moving = False
            else:
                dice = base_dice[d]
                extra = 0

                # 卡卡罗 (3)：若处于最后一名 → 额外 +3
                if d == 3:
                    min_f = 999999
                    for j in range(6):
                        if finished[j] == 0 and fwd[j] < min_f:
                            min_f = fwd[j]
                    if fwd[d] == min_f:
                        extra += 3

                # 绯雪 (4)：若已遇见哈基布 → 额外 +1
                if d == 4 and feixue_met:
                    extra += 1

                # 卡提希娅 (5)：技能解锁后 → 60% 概率额外 +2
                cartethyia_used_this_move = 0
                if d == 5 and cartethyia_unlocked:
                    if _rng_random(rng) < 0.60:
                        extra += 2
                        cartethyia_used_this_move = 1

                total_steps = dice + extra
                forward_moving = True

            # --- 计算目标格 ---
            if forward_moving:
                new_p = (p + total_steps) % 32
            else:
                new_p = (p - total_steps) % 32

            # --- 移动团子组 ---
            moved = _move_group(tile_bot, above, below, pos, p, d, new_p)

            if forward_moving:
                for k in range(len(moved)):
                    md = moved[k]
                    fwd[md] += total_steps
                    if md != 6 and pos[md] > max_reg_pos:
                        max_reg_pos = pos[md]

                # 绯雪 (4)：移动后检查是否与哈基布同格
                if not feixue_met:
                    for k in range(len(moved)):
                        if moved[k] == 4:
                            if pos[6] == pos[4]:
                                feixue_met = 1
                            break

                # 移动后检查终点
                finish_counter, win = _check_win(
                    moved, fwd, finished, rank, finish_counter)
                if win:
                    game_over = True

                # 卡提希娅 (5)：移动结束后若处于最后一名 → 解锁技能
                if d == 5 and not cartethyia_unlocked and not finished[5]:
                    min_f = 999999
                    for j in range(6):
                        if finished[j] == 0 and fwd[j] < min_f:
                            min_f = fwd[j]
                    if fwd[5] == min_f:
                        cartethyia_unlocked = 1

            if game_over:
                break

            # ---- 处理落点特殊格效果 ----
            if _is_green(new_p):
                if forward_moving:
                    _effect_green_red(tile_bot, above, below, pos, fwd,
                                      finished, moved, forward=True)
                    finish_counter, win = _check_win(
                        moved, fwd, finished, rank, finish_counter)
                    if win:
                        game_over = True
                else:
                    _effect_green_red_hakibu(tile_bot, above, below, pos,
                                             forward=True)

            elif _is_red(new_p):
                if forward_moving:
                    _effect_green_red(tile_bot, above, below, pos, fwd,
                                      finished, moved, forward=False)
                else:
                    _effect_green_red_hakibu(tile_bot, above, below, pos,
                                             forward=False)

            elif _is_black(new_p):
                _effect_black(rng, tile_bot, above, below, pos, new_p)

            if game_over:
                break

            # 哈基布相遇判定：若落点有活跃普通团子 → 哈基布沉底
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
                    _dumpling_to_bottom(tile_bot, above, below, pos, hp, 6)

                # 绯雪 (4)：哈基布移动后检查与绯雪同格
                if not feixue_met:
                    cur2 = tile_bot[hp]
                    while cur2 != -1:
                        if cur2 == 4:
                            feixue_met = 1
                            break
                        cur2 = above[cur2]

        # ===== 哈基布重置：若有普通团子位置 > 哈基布位置 → 传送回 31 =====
        if not game_over and round_num >= 4:
            hp = pos[6]
            for j in range(6):
                if not finished[j] and pos[j] > hp:
                    _dumpling_teleport(tile_bot, above, below, pos, 6, 31)
                    break

    # ---- 对未到达终点的团子排名 ----
    remaining = np.zeros(6, dtype=np.int32)
    n_rem = 0
    for d in range(6):
        if finished[d] == 0:
            remaining[n_rem] = d
            n_rem += 1

    # 按 (fwd 降序, 堆叠索引 降序) 排序：fwd 越大名次越好；同格时越靠上越好
    for i in range(n_rem):
        for j2 in range(i + 1, n_rem):
            di = remaining[i]
            dj = remaining[j2]
            hi = _stack_idx(tile_bot, above, pos[di], di)
            hj = _stack_idx(tile_bot, above, pos[dj], dj)
            if fwd[di] < fwd[dj] or (fwd[di] == fwd[dj] and hi < hj):
                remaining[i], remaining[j2] = remaining[j2], remaining[i]

    for i in range(n_rem):
        rank[remaining[i]] = finish_counter + 1 + i

    result = np.zeros(6, dtype=np.int32)
    for i in range(6):
        result[i] = rank[i]
    return result


# =============================================================================
# 批量运行（多进程 worker）
# =============================================================================

def run_batch(args: Tuple[int, int]) -> Dict[int, Dict[int, int]]:
    """
    运行 n_sims 次模拟，种子从 batch_id * 1_000_000 开始。
    返回 {团子索引: {名次: 次数}}。
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
    """将批次结果合并到全局结果字典中。"""
    for d in range(6):
        for rank_val, cnt in batch[d].items():
            results[d][rank_val] += cnt


# =============================================================================
# 统计函数
# =============================================================================

def wilson_ci(successes: int, total: int) -> Tuple[float, float, float]:
    """Wilson 置信区间（99%）。返回 (p_hat, lower, upper)。"""
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
    """格式化时间。"""
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    h, m = divmod(int(sec), 3600)
    return f"{h}h{m // 60}m"


# =============================================================================
# 详细测试模式（纯 Python 参考实现，用于调试单局比赛）
# =============================================================================

class VerboseGame:
    """纯 Python 参考实现，用于逐步输出单局比赛的详细信息。"""

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

        self.augusta_skip = False       # 奥古斯塔本回合跳过
        self.augusta_force_last = False # 奥古斯塔下回合最后行动
        self.yuno_used = False          # 尤诺技能已使用
        self.yuno_mid = False           # 尤诺已过中点
        self.feixue_met = False         # 绯雪已遇见哈基布
        self.cartethyia_unlocked = False # 卡提希娅技能已解锁

        self.base_dice = [0] * 7

        # 初始：6 个普通团子在 0 号位随机堆叠
        order = list(range(6))
        self.rng.shuffle(order)
        for d in order:
            self.board[0].append(d)
        # 哈基布在 31 号位
        self.pos[6] = 31
        self.board[31].append(6)

    def _find(self, d):
        """查找团子 d 的位置和堆叠索引。"""
        p = self.pos[d]
        try:
            idx = self.board[p].index(d)
            return p, idx
        except ValueError:
            for pp in range(NUM_POSITIONS):
                if d in self.board[pp]:
                    self.pos[d] = pp
                    return pp, self.board[pp].index(d)
        raise RuntimeError(f"团子 {d} 未找到")

    def _move_group(self, from_p, stack_idx, steps, forward):
        """将团子及其上方团子整体移动。"""
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
        """检查移动组中是否有团子完成比赛。"""
        for d in reversed(moved):  # 从上到下
            if d == 6 or self.finished[d]:
                continue
            if self.fwd[d] >= 31:
                self.finished[d] = True
                self.finish_counter += 1
                self.rank[d] = self.finish_counter
                self.game_over = True

    def _green_red(self, moved, forward):
        """绿色/红色格效果：移动团子组 1 格。"""
        for d in moved:
            if d == 6:
                continue
            if self.finished[d]:
                continue
            p = self.pos[d]
            self.board[p].remove(d)
            np_tile = (p + 1) % 32 if forward else (p - 1) % 32
            self.pos[d] = np_tile
            self.board[np_tile].append(d)
            if forward:
                self.fwd[d] += 1
            else:
                self.fwd[d] -= 1

    def _black(self, p):
        """黑色格效果：随机重排普通团子。"""
        tile = self.board[p]
        non_h = [d for d in tile if d != 6]
        h_list = [d for d in tile if d == 6]
        if len(non_h) > 1:
            self.rng.shuffle(non_h)
        self.board[p] = h_list + non_h

    def _compute_ranking(self):
        """返回活跃普通团子的当前排名（从好到差）。"""
        active = [d for d in range(6) if not self.finished[d]]
        active.sort(key=lambda d: (
            self.fwd[d],
            self._find(d)[1]
        ), reverse=True)
        return active

    def _yuno(self):
        """尤诺技能：传送排名相邻的团子。"""
        self.yuno_used = True
        target = self.pos[1]
        active = self._compute_ranking()
        yuno_idx = active.index(1)
        tele = []
        if yuno_idx > 0:
            tele.append(active[yuno_idx - 1])
        if yuno_idx < len(active) - 1:
            tele.append(active[yuno_idx + 1])
        if not tele:
            return
        for d in tele:
            self.board[self.pos[d]].remove(d)
        # 先放排名差的，再放排名好的，使较好者靠上
        for d in reversed(tele):
            self.pos[d] = target
            self.board[target].append(d)

    def run(self):
        while not self.game_over:
            self.round_num += 1
            print(f"\n{'='*50}")
            print(f"第 {self.round_num} 回合")
            print(f"{'='*50}")

            # 预生成骰子
            for d in range(6):
                if not self.finished[d]:
                    self.base_dice[d] = self.rng.randint(1, 3)
            if self.round_num >= 4:
                self.base_dice[6] = self.rng.randint(1, 6)

            # 回合开始前技能判定
            for d in range(6):
                if self.finished[d]:
                    continue
                p, sidx = self._find(d)
                tile = self.board[p]
                is_top = (sidx == len(tile) - 1)

                if d == 0 and is_top:  # 奥古斯塔
                    self.augusta_skip = True
                    self.augusta_force_last = True

                if d == 2 and sidx < len(tile) - 1:  # 今汐
                    if self.rng.random() < 0.40:
                        tile.remove(d)
                        tile.append(d)

                if d == 1 and not self.yuno_used:  # 尤诺
                    if self.fwd[d] >= 16:
                        self.yuno_mid = True
                    if self.yuno_mid:
                        if any(not self.finished[j] for j in range(6) if j != 1):
                            self._yuno()

            # 构建行动顺序
            active = [i for i in range(6) if not self.finished[i]]
            if self.round_num >= 4:
                active.append(6)
            self.rng.shuffle(active)

            normal, last_g = [], []
            for d in active:
                if d == 0 and self.augusta_skip:
                    self.augusta_skip = False
                    continue
                if self.augusta_force_last and d == 0:
                    last_g.append(d)
                    self.augusta_force_last = False
                else:
                    normal.append(d)
            self.rng.shuffle(last_g)
            order = normal + last_g

            names = [DUMPLING_NAMES[d] if d < 6 else '哈基布' for d in order]
            print(f"行动顺序: {names}")

            max_reg_pos = -1
            for d in order:
                if self.game_over or self.finished[d]:
                    continue
                name = DUMPLING_NAMES[d] if d < 6 else '哈基布'
                p, sidx = self._find(d)

                if d == 6:
                    dice = self.base_dice[6]
                    new_p, moved = self._move_group(p, sidx, dice, False)
                    print(f"  哈基布 在 {p} 号位 投 {dice} 点 -> {new_p} 号位")
                    if _is_green.py_func(new_p):
                        self._move_group(new_p, self.board[new_p].index(6), 1, True)
                        print(f"    绿色格！-> {self.pos[6]} 号位")
                    elif _is_red.py_func(new_p):
                        self._move_group(new_p, self.board[new_p].index(6), 1, False)
                        print(f"    红色格！-> {self.pos[6]} 号位")
                    elif _is_black.py_func(new_p):
                        self._black(new_p)
                        print(f"    黑色格！重新堆叠")
                    # 相遇判定
                    hp = self.pos[6]
                    has_active = any(d2 != 6 and not self.finished[d2] for d2 in self.board[hp])
                    if has_active:
                        self.board[hp].remove(6)
                        self.board[hp].insert(0, 6)
                        print(f"    相遇！哈基布沉底")
                    if not self.feixue_met and 4 in self.board[hp]:
                        self.feixue_met = True
                        print(f"    绯雪遇见哈基布！")
                else:
                    dice = self.base_dice[d]
                    extra = 0
                    if d == 3:  # 卡卡罗
                        min_f = min(self.fwd[j] for j in range(6) if not self.finished[j])
                        if self.fwd[d] == min_f:
                            extra += 3
                    if d == 4 and self.feixue_met:  # 绯雪
                        extra += 1
                    if d == 5 and self.cartethyia_unlocked:  # 卡提希娅
                        if self.rng.random() < 0.60:
                            extra += 2
                    steps = dice + extra
                    new_p, moved = self._move_group(p, sidx, steps, True)
                    extra_str = f"+{extra}" if extra else ""
                    print(f"  {name} 在 {p} 号位 投 {dice}{extra_str} 点 -> {new_p} 号位")
                    for m in moved:
                        if m != 6 and self.pos[m] > max_reg_pos:
                            max_reg_pos = self.pos[m]

                    self._check_win(moved)
                    if self.game_over:
                        print(f"    >>> 有人抵达终点！游戏结束 <<<")

                    if not self.game_over:
                        if _is_green.py_func(new_p):
                            self._green_red(moved, True)
                            print(f"    绿色格！-> {[self.pos[m] for m in moved]} 号位")
                            self._check_win(moved)
                        elif _is_red.py_func(new_p):
                            self._green_red(moved, False)
                            print(f"    红色格！-> {[self.pos[m] for m in moved]} 号位")
                        elif _is_black.py_func(new_p):
                            self._black(new_p)
                            print(f"    黑色格！重新堆叠")

                    if not self.feixue_met and 4 in moved:
                        if self.pos[6] == self.pos[4]:
                            self.feixue_met = True
                            print(f"    绯雪遇见哈基布！")

                    if d == 5 and not self.cartethyia_unlocked and not self.finished[5]:
                        min_f = min(self.fwd[j] for j in range(6) if not self.finished[j])
                        if self.fwd[5] == min_f:
                            self.cartethyia_unlocked = True
                            print(f"    卡提希娅技能解锁！")

                if self.game_over:
                    break

            if not self.game_over and self.round_num >= 4:
                hp = self.pos[6]
                if any(not self.finished[j] and self.pos[j] > hp for j in range(6)):
                    print(f"  哈基布传送回 31 号位")
                    self.board[hp].remove(6)
                    self.pos[6] = 31
                    self.board[31].append(6)

            print(f"\n  第 {self.round_num} 回合结束后棋盘:")
            for pp in range(NUM_POSITIONS):
                if self.board[pp]:
                    tile_names = [DUMPLING_NAMES[d2] if d2 < 6 else '哈基ブ' for d2 in self.board[pp]]
                    print(f"    {pp} 号位: {tile_names}")

        # 对未到达终点的团子排名
        remaining = [d for d in range(6) if not self.finished[d]]
        remaining.sort(key=lambda d: (
            self.fwd[d],
            self._find(d)[1]
        ), reverse=True)
        for i, d in enumerate(remaining):
            self.rank[d] = self.finish_counter + 1 + i
        return self.rank[:6]


# =============================================================================
# 主函数
# =============================================================================

def main():
    # Windows 下确保 UTF-8 输出
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    import argparse

    parser = argparse.ArgumentParser(
        description='团子赛跑 Monte Carlo 模拟 — Group A（第二版，numba 优化）')
    parser.add_argument('sims', nargs='?', type=int, default=1_000_000,
                        help='模拟次数（默认: 1,000,000）')
    parser.add_argument('--workers', '-w', type=int, default=max(1, cpu_count() - 1),
                        help=f'并行进程数（默认: cpu_count-1 = {max(1, cpu_count()-1)}）')
    parser.add_argument('--batch', '-b', type=int, default=50000,
                        help='每个 worker 任务的批次大小（默认: 50000）')
    parser.add_argument('--test', '-t', action='store_true',
                        help='运行单局详细测试以调试')
    parser.add_argument('--test-n', type=int, default=1,
                        help='详细测试的运行次数')
    args = parser.parse_args()

    # --- 测试模式 ---
    if args.test:
        print("=" * 50)
        print("  详细测试模式 — Group A（第二版）")
        print("=" * 50)
        for i in range(args.test_n):
            seed = i * 1000
            print(f"\n\n>>> 第 {i+1} 局 (seed={seed}) <<<")
            g = VerboseGame(seed)
            ranks = g.run()
            print(f"\n  最终排名:")
            for d in sorted(range(6), key=lambda x: ranks[x]):
                print(f"    第 {ranks[d]} 名: {DUMPLING_NAMES[d]}")
        return

    # --- 生产运行 ---
    TOTAL = args.sims
    WORKERS = args.workers
    BATCH = args.batch

    print("=" * 70)
    print("  团子赛跑 — Group A（第二版）Monte Carlo 模拟")
    print("  引擎: numba JIT + 链表棋盘 + 多进程")
    print("=" * 70)
    print(f"  模拟次数: {TOTAL:,}")
    print(f"  并行进程: {WORKERS}")
    print(f"  批次大小: {BATCH:,}")
    print(f"  置信水平: 99%（Wilson 置信区间）")
    print()
    print("  对歧义规则的解释：")
    print("    1. 终点判定: linear distance >= 31（允许越过 31 号点）")
    print("    2. 特殊格: 不连锁触发")
    print("    3. 哈基布从第 4 回合起加入行动序列")
    print("    4. 奥古斯塔\"最后行动\" = 整个序列最后（含哈基布之后）")
    print("    5. 尤诺: 传送排名相邻的团子；排名较高者堆叠靠上")
    print("    6. 卡提希娅: +2 格加在基础骰子结果之上")
    print("=" * 70)
    print()

    # 预热：强制 numba JIT 编译
    print("  正在编译模拟引擎（numba JIT）...", end=" ", flush=True)
    _ = simulate_one_game(0)
    print("完成。\n")

    # 构建批次列表
    full_batches = TOTAL // BATCH
    remainder = TOTAL % BATCH
    batches = [(i, BATCH) for i in range(full_batches)]
    if remainder:
        batches.append((full_batches, remainder))

    print(f"  总批次数: {len(batches)}")
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
                      f"{rate:,.0f} 局/秒  "
                      f"预计剩余 {fmt_dur(eta)}")

    except KeyboardInterrupt:
        print("\n  已中断。基于已完成的部分数据计算结果...")
        TOTAL = max(1, completed)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  模拟结果（{TOTAL:,} 局，耗时 {fmt_dur(elapsed)}）")
    print(f"{'='*70}")

    # --- 胜率与 99% 置信区间 ---
    header = f"  {'团子':<10} {'胜率':>10}  {'99% CI':>24}  {'获胜次数':>10}"
    print(header)
    print(f"  {'-'*10} {'-'*10}  {'-'*24}  {'-'*10}")

    for d in range(6):
        wins = results[d].get(1, 0)
        p, lo, hi = wilson_ci(wins, TOTAL)
        ci_str = f"[{100*lo:.4f}%, {100*hi:.4f}%]"
        print(f"  {DUMPLING_NAMES[d]:<10} {100*p:>9.4f}%  {ci_str:>24}  {wins:>10,}")

    # --- 名次分布 ---
    print(f"\n  名次分布:")
    print(f"  {'团子':<10}", end="")
    for r in range(1, 7):
        print(f"  第{r}名   ", end="")
    print(f"\n  {'-'*10}", end="")
    for _ in range(6):
        print(f"  ------", end="")
    print()
    for d in range(6):
        print(f"  {DUMPLING_NAMES[d]:<10}", end="")
        for r in range(1, 7):
            cnt = results[d].get(r, 0)
            print(f"  {100*cnt/TOTAL:5.2f}%", end="")
        print()

    # --- 前2名 / 前3名概率 ---
    print(f"\n  前2名 / 前3名 概率:")
    print(f"  {'团子':<10} {'前2名':>10}  {'99% CI':>24}  {'前3名':>10}  {'99% CI':>24}")
    print(f"  {'-'*10} {'-'*10}  {'-'*24}  {'-'*10}  {'-'*24}")
    for d in range(6):
        top2 = sum(results[d].get(r, 0) for r in (1, 2))
        top3 = sum(results[d].get(r, 0) for r in (1, 2, 3))
        p2, lo2, hi2 = wilson_ci(top2, TOTAL)
        p3, lo3, hi3 = wilson_ci(top3, TOTAL)
        ci2_str = f"[{100*lo2:.4f}%, {100*hi2:.4f}%]"
        ci3_str = f"[{100*lo3:.4f}%, {100*hi3:.4f}%]"
        print(f"  {DUMPLING_NAMES[d]:<10} {100*p2:>9.4f}%  {ci2_str:>24}  {100*p3:>9.4f}%  {ci3_str:>24}")

    # --- 平均名次 ---
    print(f"\n  平均名次:")
    for d in range(6):
        avg = sum(r * cnt for r, cnt in results[d].items()) / TOTAL
        print(f"  {DUMPLING_NAMES[d]:<10} {avg:.4f}")

    # --- 验证 ---
    total_win = sum(results[d].get(1, 0) for d in range(6))
    print(f"\n  胜率之和: {100*total_win/TOTAL:.4f}%  "
          f"{'(应接近 100%)' if abs(total_win/TOTAL - 1.0) < 0.002 else '警告: 请检查逻辑!'}")

    print(f"\n{'='*70}")
    print("  运行完成。")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
