"""
Poisson + Dixon-Coles match prediction engine.

Improvements over basic Poisson:
  1. Dixon-Coles τ correction — adjusts 0-0, 1-0, 0-1, 1-1 probabilities
     which pure Poisson systematically mis-estimates.
  2. League-specific xG averages and home-advantage multipliers.
  3. Score-matrix capped at MAX_GOALS to keep computation fast.
"""

import math
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_GOALS = 9   # consider 0..8 goals per team

# Dixon-Coles correlation parameter (standard estimate from their 1997 paper)
RHO = -0.13

# League-specific parameters: (home_avg_xG, away_avg_xG, home_advantage_multiplier)
# home_advantage_multiplier is already baked into home_avg but kept separate
# for transparency. Neutral venues (World Cup) get 1.0.
LEAGUE_PARAMS: dict[int, dict] = {
    2000: {"home_avg": 1.22, "away_avg": 1.22, "home_adv": 1.00},  # FIFA World Cup (neutral)
    2001: {"home_avg": 1.52, "away_avg": 1.18, "home_adv": 1.15},  # UEFA Champions League
    2021: {"home_avg": 1.55, "away_avg": 1.15, "home_adv": 1.18},  # Premier League
    2014: {"home_avg": 1.45, "away_avg": 1.10, "home_adv": 1.15},  # La Liga
    2002: {"home_avg": 1.62, "away_avg": 1.28, "home_adv": 1.18},  # Bundesliga (high scoring)
    2019: {"home_avg": 1.32, "away_avg": 1.05, "home_adv": 1.12},  # Serie A (defensive)
    2015: {"home_avg": 1.42, "away_avg": 1.10, "home_adv": 1.20},  # Ligue 1
    2013: {"home_avg": 1.50, "away_avg": 1.15, "home_adv": 1.25},  # Brasileiro Série A
    2152: {"home_avg": 1.38, "away_avg": 1.05, "home_adv": 1.28},  # Copa Libertadores
    2016: {"home_avg": 1.45, "away_avg": 1.15, "home_adv": 1.15},  # Championship
    2003: {"home_avg": 1.58, "away_avg": 1.22, "home_adv": 1.18},  # Eredivisie
    2017: {"home_avg": 1.38, "away_avg": 1.05, "home_adv": 1.15},  # Primeira Liga
}

_DEFAULT_PARAMS = {"home_avg": 1.45, "away_avg": 1.10, "home_adv": 1.15}


# ── Core maths ────────────────────────────────────────────────────────────────

def _poisson(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def _tau(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Dixon-Coles low-score correction factor.
    Adjusts the independence assumption for scorelines where i+j <= 1
    and the exact 1-1 draw.
    """
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    elif i == 1 and j == 0:
        return 1.0 + lam_a * rho
    elif i == 0 and j == 1:
        return 1.0 + lam_h * rho
    elif i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _score_matrix(lam_h: float, lam_a: float) -> list[list[float]]:
    """MAX_GOALS × MAX_GOALS matrix of P(home=i, away=j) with DC correction."""
    matrix = []
    for i in range(MAX_GOALS):
        row = []
        for j in range(MAX_GOALS):
            p = _poisson(lam_h, i) * _poisson(lam_a, j) * _tau(i, j, lam_h, lam_a, RHO)
            row.append(max(p, 0.0))   # guard against tiny negatives from τ
        matrix.append(row)
    return matrix


# ── Public interface ──────────────────────────────────────────────────────────

def predict(lam_home: float, lam_away: float) -> dict[str, Any]:
    """Full prediction from two expected-goals rates."""
    m = _score_matrix(lam_home, lam_away)

    home_win = sum(m[i][j] for i in range(MAX_GOALS) for j in range(i))
    draw     = sum(m[i][i] for i in range(MAX_GOALS))
    away_win = sum(m[i][j] for j in range(MAX_GOALS) for i in range(j))

    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw     /= total
        away_win /= total

    btts   = (1 - _poisson(lam_home, 0)) * (1 - _poisson(lam_away, 0))
    over25 = 1.0 - sum(
        m[i][j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j <= 2
    )

    # Most-likely scoreline
    best_prob, best_h, best_a = 0.0, 1, 0
    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            if m[i][j] > best_prob:
                best_prob, best_h, best_a = m[i][j], i, j

    if home_win >= draw and home_win >= away_win:
        winner, confidence = "home", home_win
    elif away_win >= draw and away_win >= home_win:
        winner, confidence = "away", away_win
    else:
        winner, confidence = "draw", draw

    return {
        "home_win_prob":    round(home_win, 3),
        "draw_prob":        round(draw, 3),
        "away_win_prob":    round(away_win, 3),
        "btts_prob":        round(btts, 3),
        "over_25_prob":     round(over25, 3),
        "predicted_winner": winner,
        "confidence":       round(confidence, 3),
        "home_xg":          round(lam_home, 2),
        "away_xg":          round(lam_away, 2),
        "predicted_score":  f"{best_h}-{best_a}",
    }


def predict_from_strengths(
    home_attack:  float,
    home_defense: float,
    away_attack:  float,
    away_defense: float,
    league_id:    int = 0,
) -> dict[str, Any]:
    """
    Compute xG rates from team strength multipliers and league context,
    then run the full Dixon-Coles Poisson prediction.

    attack  > 1.0 → scores more than average
    defense < 1.0 → concedes less (strong defence)

    xG_home = home_avg * home_attack * away_defense * home_advantage
    xG_away = away_avg * away_attack * home_defense
    """
    p = LEAGUE_PARAMS.get(league_id, _DEFAULT_PARAMS)

    lam_h = p["home_avg"] * home_attack  * away_defense * p["home_adv"]
    lam_a = p["away_avg"] * away_attack  * home_defense

    # Clip to sensible range (prevents degenerate probabilities)
    lam_h = max(0.3, min(lam_h, 5.0))
    lam_a = max(0.3, min(lam_a, 5.0))

    return predict(lam_h, lam_a)
