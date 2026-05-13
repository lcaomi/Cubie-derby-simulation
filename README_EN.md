# 2026 2nd Solaris——Cubie Derby — Monte Carlo Simulation

中文文档请点击:[README.md](https://github.com/lcaomi/Cubie-derby-simulation/README.md)

<p align="center">
  <img
    width="240"
    height="240"
    alt="image"
    src="[https://github.com/user-attachments/assets/944c8a50-f463-4040-b1d2-625560db22bc](https://github.com/user-attachments/assets/7dfada9c-304b-4c6d-b07c-f990f8b989ff)"
</p>


## ⚠️ Disclaimer ⚠️
**This project is based on the event "Cubie Derby" map from KURO GAMES version 3.3. All text, skill descriptions, and map explanations belong to KURO GAMES. The author assumes no responsibility for any results. This project is for entertainment and practice purposes only and should not be used as any valid reference or recommendation.**

**My doughter and Wife Play whatever you like.**

---

## Technical Approach
This project uses **Numba JIT + multiprocessing** to perform large-scale Monte Carlo simulations.

- Win probabilities
- Expected rankings
- 99% confidence intervals (Wilson score interval)

The project includes **Group B** and **Group C** implementations.

---

## Project Structure
```
Cubie-derby-simulation/
├── CubieDerby_Gp.B-A/
│   ├── simulate.py
│   ├── For_Claude.txt
│   └── README.md
├── CubieDerby_Gp.C-A/
│   ├── simulation.py
│   └── additional.txt
├── Require.txt
└── README.md
```

---

## Group B vs Group C

### Cubies
- **Group B:** Qianxiao, Morning, Linna, Aimis, Guardian, Coletta
- **Group C:** Augusta, Yuno, Fnono, Changli, Jinxi, Calcharo

### Game End
- Ends immediately when any cubie reaches **tile 31**

### Hakib Action
- Activated starting from **Round 4**

### Dice Generation
- Base dice pre-generated each round

### Ranking Rules
1. Position (closer to 31 is better)
2. Stack height

---

## Game Rules Overview

### Map
- 32 tiles (0–31), circular
- Normal cubies: forward (0→31)
- Hakib: backward (31→0)

| Type | Positions | Effect |
|------|----------|--------|
| Green | 2, 10, 15, 22 | Move +1 |
| Red   | 9, 27 | Move -1 |
| Black | 5, 19 | Shuffle normal cubies on tile |

---

### General Rules
- Normal cubies roll **1–3**
- Carry cubies above when moving
- Arriving group placed on **top**
- Hakib rolls **1–6**, stacks **bottom**
- Effects do **not chain**
- Higher stack ranks higher
- Hakib excluded from ranking

---

## Group B Skills

| Cubie | Skill |
|------|------|
| Qianxiao | If smallest dice → +2 steps |
| Morning | Dice cycle 3→2→1 |
| Linna | 60% double / 20% no move / 20% normal |
| Aimis | After tile 15 → teleport (once) |
| Guardian | Only rolls 2 or 3 |
| Coletta | 28% double chance |

---

## Group C Skills

| Cubie | Skill |
|------|------|
| Augusta | Skip turn if top; next round last |
| Yuno | Teleport others after midpoint |
| Fnono | Bottom → +3 |
| Changli | 65% last if someone below |
| Jinxi | 40% move to top |
| Calcharo | Last place → +3 |

---

## Technical Implementation
- Numba JIT (`@njit`)
- Linked-list board structure
- SplitMix64 RNG
- Multiprocessing
- Wilson interval

---

## Quick Start

### Group B
```bash
cd CubieDerby_Gp.B-A
python simulate.py --trials 10000000 --seed 42
```

### Group C
```bash
cd CubieDerby_Gp.C-A
python simulation.py 10000000 --workers 8
```

---

## Parameters

| Parameter | Description |
|----------|-------------|
| --trials | Number of simulations |
| --seed | Random seed |
| --workers | Processes |
| --batch | Batch size |

---

## Assumptions

| # | Issue | Assumption |
|--|------|-----------|
| 1 | Hakib start | Round 4 |
| 2 | Chain effects | Disabled |
| 3 | Over 31 | B: stop, C: loop |
| 4 | Black tile | normal only |
| 5 | Carry Hakib | No |

---

## Dependencies
```bash
pip install numpy numba
```

---
