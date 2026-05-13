# Cubie Derby — 2026 Monte Carlo Simulation

<p align="center">
  <img
    width="240"
    height="240"
    alt="0f83c661059d08570ab9cfa032a91b59_720"
    src="https://github.com/user-attachments/assets/944c8a50-f463-4040-b1d2-625560db22bc"
</p>

## ⚠️ Disclaimer ⚠️
This project is based on the Cubie Derby map from KURO GAMES version 3.3. All text, skill descriptions, and map explanations belong to KURO GAMES. The author assumes no responsibility for any results. This project is for entertainment and practice purposes only.

---

## Technical Approach
Large-scale Monte Carlo simulation using Numba JIT and multiprocessing.

- Win probabilities
- Expected rankings
- 99% confidence intervals

---

## Game Rules Overview
- 32 positions (0–31), loop map
- Normal cubies move forward
- Hakib moves backward

---

## Quick Start
```bash
cd CubieDerby_Gp.B-A
python simulate.py --trials 10000000 --seed 42
```

```bash
cd CubieDerby_Gp.C-A
python simulation.py 10000000 --workers 8
```

---

## Dependencies
```bash
pip install numpy numba
```

---

*Generated English translation markdown file*
