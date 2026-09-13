"""Dixon-Coles bivariate Poisson scoreline model.

Adapted from the implementation in the monorepo's artifacts/predictor-api,
which applies Dixon & Coles (1997). Two reasons it is worth adopting here:

1. Independent Poisson marginals systematically misprice low scorelines -
   0-0, 1-0, 0-1 and 1-1 are correlated in real football. The tau correction
   adjusts exactly those four cells.
2. Every market is read off one joint distribution, so 1X2, BTTS, any goal
   line and correct score are mutually consistent by construction. Deriving
   them separately lets them contradict each other - a set of "probabilities"
   that cannot all be true at once.

The difference from the monorepo version is where the expected goals come
from. There they are computed from a hand-maintained table of team ratings;
here they come from the rolling attack and defence rates the feature pipeline
already learns from results, so they update themselves and can be validated
against outcomes.
"""

from math import exp, factorial
from typing import Any

# Scorelines above this contribute negligible probability mass.
MAX_GOALS = 9

# Low-score correlation. Dixon & Coles estimate rho near -0.13 across English
# league data; it is a fixed prior here rather than fitted per league.
RHO = -0.13

# Guards against degenerate rates from extreme or sparse inputs.
MIN_RATE = 0.3
MAX_RATE = 5.0


def _poisson(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam**k) * exp(-lam) / factorial(k)


def _tau(i: int, j: int, lam_h: float, lam_a: float, rho: float = RHO) -> float:
    """Dixon-Coles correction, applied only to the four low scorelines."""
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam_home: float, lam_away: float) -> list[list[float]]:
    """Joint distribution over scorelines, normalised to sum to 1."""
    lam_home = max(MIN_RATE, min(lam_home, MAX_RATE))
    lam_away = max(MIN_RATE, min(lam_away, MAX_RATE))

    matrix = [
        [
            max(_poisson(lam_home, i) * _poisson(lam_away, j) * _tau(i, j, lam_home, lam_away), 0.0)
            for j in range(MAX_GOALS)
        ]
        for i in range(MAX_GOALS)
    ]
    # Truncation at MAX_GOALS and the tau clamp both lose a little mass.
    total = sum(sum(row) for row in matrix)
    if total > 0:
        matrix = [[cell / total for cell in row] for row in matrix]
    return matrix


def outcomes(matrix: list[list[float]]) -> dict[str, Any]:
    """Read every market off one joint distribution."""
    home_win = sum(matrix[i][j] for i in range(MAX_GOALS) for j in range(i))
    draw = sum(matrix[i][i] for i in range(MAX_GOALS))
    away_win = sum(matrix[i][j] for j in range(MAX_GOALS) for i in range(j))
    btts = sum(matrix[i][j] for i in range(1, MAX_GOALS) for j in range(1, MAX_GOALS))

    best = max(
        ((i, j, matrix[i][j]) for i in range(MAX_GOALS) for j in range(MAX_GOALS)),
        key=lambda cell: cell[2],
    )
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "btts": btts,
        "predicted_score": f"{best[0]}-{best[1]}",
    }


def over_line(matrix: list[list[float]], line: float) -> float:
    """P(total goals > line) for any half-goal line, from the same matrix."""
    return sum(
        matrix[i][j]
        for i in range(MAX_GOALS)
        for j in range(MAX_GOALS)
        if i + j > line
    )


# Goals are not evenly spread across a match: roughly 45% arrive in the first
# half and 55% in the second, consistently across major leagues. Splitting the
# match rate this way is what makes half-based markets derivable at all.
FIRST_HALF_SHARE = 0.45


def half_rates(lam_home: float, lam_away: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Split match expected goals into first- and second-half rates."""
    first = (lam_home * FIRST_HALF_SHARE, lam_away * FIRST_HALF_SHARE)
    second = (lam_home * (1 - FIRST_HALF_SHARE), lam_away * (1 - FIRST_HALF_SHARE))
    return first, second


def win_either_half(lam_home: float, lam_away: float, side: str = "home") -> float:
    """P(the side wins at least one half, scored as two separate matches).

    Halves are treated as independent given their rates. They are not strictly
    independent - a team leading at the break often changes how it plays - so
    this is an approximation, and is labelled derived wherever it surfaces.
    """
    (h1_home, h1_away), (h2_home, h2_away) = half_rates(lam_home, lam_away)

    def win_probability(home_rate: float, away_rate: float) -> float:
        result = outcomes(score_matrix(home_rate, away_rate))
        return result["home_win"] if side == "home" else result["away_win"]

    p_first = win_probability(h1_home, h1_away)
    p_second = win_probability(h2_home, h2_away)
    return 1 - (1 - p_first) * (1 - p_second)


def first_half(lam_home: float, lam_away: float) -> dict[str, Any]:
    """Markets on the first half alone, scored as its own short match."""
    (h1_home, h1_away), _ = half_rates(lam_home, lam_away)
    matrix = score_matrix(h1_home, h1_away)
    result = outcomes(matrix)
    return {
        "home_win": result["home_win"],
        "draw": result["draw"],
        "away_win": result["away_win"],
        "over_0_5": over_line(matrix, 0.5),
        "over_1_5": over_line(matrix, 1.5),
        "expected_goals": round(h1_home + h1_away, 2),
    }


def handicap(lam_home: float, lam_away: float, line: float, side: str = "home") -> float:
    """P(side covers the handicap), read off the same scoreline distribution.

    A handicap adds goals to one side before comparing. Only half lines are
    supported: whole-number lines can end level and push, which returns the
    stake rather than winning, and a single probability cannot express that.
    """
    if float(line).is_integer():
        raise ValueError("Whole-number handicaps can push; use a half line such as -1.5.")

    matrix = score_matrix(lam_home, lam_away)
    covered = 0.0
    for home_goals in range(MAX_GOALS):
        for away_goals in range(MAX_GOALS):
            if side == "home":
                if home_goals + line > away_goals:
                    covered += matrix[home_goals][away_goals]
            elif away_goals + line > home_goals:
                covered += matrix[home_goals][away_goals]
    return covered


def expected_goals(features: dict[str, float]) -> tuple[float, float]:
    """Expected goals from learned rolling rates.

    Each side's scoring rate is blended with the opponent's concession rate,
    both of which the feature pipeline maintains as rolling averages over
    previous fixtures only.
    """
    home = (features["home_goals_scored_avg"] + features["away_goals_conceded_avg"]) / 2
    away = (features["away_goals_scored_avg"] + features["home_goals_conceded_avg"]) / 2
    # Home advantage, consistent with the ~1.55 vs 1.15 points-per-game split
    # seen across the leagues in the training data.
    return home * 1.08, away * 0.94


def predict(features: dict[str, float]) -> dict[str, Any]:
    """Full scoreline-model output for one fixture."""
    lam_home, lam_away = expected_goals(features)
    matrix = score_matrix(lam_home, lam_away)
    result = outcomes(matrix)
    return {
        **result,
        "home_xg": round(lam_home, 2),
        "away_xg": round(lam_away, 2),
        "over_1_5": over_line(matrix, 1.5),
        "over_2_5": over_line(matrix, 2.5),
        "over_3_5": over_line(matrix, 3.5),
    }
