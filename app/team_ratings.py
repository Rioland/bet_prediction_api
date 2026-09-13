"""
Team strength database — attack/defence multipliers relative to league average.

attack  > 1.0 → scores more than average
defence < 1.0 → concedes less than average (strong defence)

Expected goals formula (in football_api.py):
  xG_home = league_home_avg * home_attack * away_defence * home_advantage
  xG_away = league_away_avg * away_attack * home_defence

Ratings are based on FIFA/ELO rankings and recent season performance.
Unknown teams fall back to a conservative (1.0, 1.0) with minor noise.
"""

from __future__ import annotations

import random

# ── Primary lookup: football-data.org team ID → (attack, defence) ────────────
RATINGS_BY_ID: dict[int, tuple[float, float]] = {
    # ── National teams (FIFA World Cup) ──────────────────────────────────────
    762: (1.60, 0.68),   # Argentina  – defending champion
    771: (1.55, 0.70),   # Brazil
    773: (1.50, 0.72),   # France
    760: (1.45, 0.72),   # Spain
    770: (1.38, 0.76),   # England
    759: (1.42, 0.74),   # Germany
    765: (1.40, 0.75),   # Portugal
    779: (1.32, 0.80),   # Netherlands
    805: (1.28, 0.80),   # Belgium
    784: (1.22, 0.82),   # Italy
    769: (1.28, 0.78),   # Uruguay
    780: (1.25, 0.82),   # Colombia
    764: (1.18, 0.85),   # Mexico
    772: (1.20, 0.82),   # Croatia
    788: (1.18, 0.83),   # Switzerland
    782: (1.15, 0.85),   # Denmark
    791: (1.12, 0.87),   # Poland
    777: (1.15, 0.85),   # Serbia
    793: (1.15, 0.84),   # Morocco
    766: (1.12, 0.87),   # Senegal
    827: (1.10, 0.87),   # Japan
    783: (1.08, 0.90),   # Ecuador
    768: (1.12, 0.88),   # USA
    785: (1.05, 0.92),   # Canada
    801: (1.05, 0.92),   # Australia
    809: (1.02, 0.93),   # South Korea
    817: (1.00, 0.95),   # Ghana
    810: (1.02, 0.93),   # Cameroon
    815: (1.08, 0.90),   # Nigeria
    763: (1.05, 0.92),   # Bolivia
    775: (1.08, 0.88),   # Paraguay
    776: (1.05, 0.92),   # Peru
    781: (1.00, 0.96),   # Venezuela
    786: (1.00, 0.96),   # Tunisia
    792: (1.02, 0.94),   # Algeria
    789: (0.98, 0.97),   # Ivory Coast
    798: (1.00, 0.95),   # Egypt
    818: (0.98, 0.97),   # Saudi Arabia
    794: (0.97, 0.98),   # Iran
    820: (0.95, 0.99),   # Qatar

    # ── Premier League ────────────────────────────────────────────────────────
    57:  (1.32, 0.78),   # Arsenal
    64:  (1.40, 0.76),   # Liverpool
    65:  (1.48, 0.70),   # Manchester City
    66:  (1.08, 0.96),   # Manchester Utd
    73:  (1.18, 0.92),   # Tottenham
    397: (1.22, 0.85),   # Aston Villa
    67:  (1.15, 0.86),   # Newcastle Utd
    61:  (1.12, 0.90),   # Chelsea
    563: (1.05, 0.95),   # West Ham
    346: (1.02, 0.97),   # Leicester City
    402: (1.00, 0.98),   # Brentford
    328: (0.98, 1.00),   # Crystal Palace
    57:  (1.32, 0.78),   # Arsenal (duplicate guard)
    76:  (0.95, 1.02),   # Wolverhampton
    341: (0.95, 1.02),   # Nottm Forest
    340: (0.98, 1.00),   # Southampton
    62:  (0.92, 1.05),   # Everton
    715: (0.90, 1.06),   # Luton Town
    354: (0.93, 1.04),   # Brighton
    338: (0.95, 1.02),   # Bournemouth
    332: (0.90, 1.06),   # Fulham
    74:  (0.88, 1.08),   # Burnley

    # ── La Liga ───────────────────────────────────────────────────────────────
    86:  (1.55, 0.68),   # Real Madrid
    81:  (1.45, 0.72),   # Barcelona
    78:  (1.15, 0.72),   # Atlético Madrid
    94:  (0.98, 1.00),   # Valencia
    90:  (1.10, 0.90),   # Real Betis
    92:  (1.12, 0.88),   # Real Sociedad
    95:  (1.05, 0.92),   # Villarreal
    82:  (1.00, 0.96),   # Athletic Club
    87:  (0.98, 0.98),   # Sevilla
    91:  (0.95, 1.02),   # Celta Vigo
    97:  (0.92, 1.05),   # Getafe
    264: (0.90, 1.06),   # Deportivo Alavés
    89:  (0.95, 1.02),   # Rayo Vallecano
    96:  (0.92, 1.05),   # Cádiz

    # ── Bundesliga ────────────────────────────────────────────────────────────
    5:   (1.58, 0.65),   # Bayern München
    4:   (1.30, 0.82),   # Borussia Dortmund
    3:   (1.35, 0.72),   # Bayer Leverkusen
    721: (1.22, 0.80),   # RB Leipzig
    19:  (1.10, 0.90),   # Eintracht Frankfurt
    11:  (1.00, 0.98),   # Wolfsburg
    18:  (1.05, 0.94),   # Borussia Mönchengladbach
    9:   (1.08, 0.92),   # Hoffenheim
    1:   (0.98, 1.00),   # FC Augsburg
    16:  (0.95, 1.02),   # VfB Stuttgart
    6:   (0.92, 1.04),   # FC Köln
    10:  (0.90, 1.06),   # Mainz 05
    43:  (0.88, 1.08),   # FC Bochum
    36:  (0.88, 1.08),   # FC Heidenheim
    12:  (0.90, 1.06),   # SC Freiburg
    28:  (0.93, 1.03),   # Werder Bremen
    29:  (0.92, 1.04),   # Hamburger SV

    # ── Serie A ───────────────────────────────────────────────────────────────
    109: (1.18, 0.82),   # Juventus
    98:  (1.22, 0.80),   # AC Milan
    108: (1.38, 0.72),   # Inter
    100: (1.12, 0.88),   # AS Roma
    113: (1.32, 0.76),   # Napoli
    110: (1.10, 0.88),   # Lazio
    107: (1.15, 0.85),   # Atalanta
    104: (1.00, 0.96),   # Fiorentina
    102: (0.98, 0.98),   # Bologna
    103: (0.95, 1.02),   # Torino
    99:  (0.92, 1.04),   # Cagliari
    101: (0.90, 1.06),   # Genoa
    106: (0.88, 1.08),   # Udinese
    105: (0.85, 1.10),   # Empoli
    112: (0.88, 1.08),   # Monza
    111: (0.90, 1.06),   # Lecce
    114: (0.85, 1.10),   # Frosinone
    116: (0.88, 1.08),   # Sassuolo
    117: (0.85, 1.10),   # Salernitana
    118: (0.83, 1.12),   # Spezia

    # ── Ligue 1 ───────────────────────────────────────────────────────────────
    524: (1.62, 0.62),   # Paris Saint-Germain
    516: (1.15, 0.88),   # Marseille
    523: (1.12, 0.90),   # Olympique Lyonnais
    548: (1.18, 0.84),   # Monaco
    511: (1.02, 0.96),   # Rennes
    512: (0.95, 1.02),   # Nantes
    514: (1.05, 0.93),   # Lille
    518: (1.00, 0.97),   # Nice
    531: (0.98, 0.99),   # RC Lens
    521: (0.95, 1.02),   # Montpellier
    519: (0.92, 1.04),   # Metz
    532: (0.92, 1.04),   # Reims
    517: (0.90, 1.06),   # Lorient
    529: (0.88, 1.08),   # Clermont Foot
    537: (0.90, 1.06),   # Le Havre
    549: (0.88, 1.08),   # Strasbourg
    546: (0.85, 1.10),   # Brest

    # ── UEFA Champions League (clubs already listed above) ───────────────────

    # ── Campeonato Brasileiro Série A ─────────────────────────────────────────
    1765: (1.28, 0.78),  # Fluminense FC
    1763: (1.38, 0.72),  # CR Flamengo
    5981: (1.35, 0.72),  # Palmeiras
    1766: (1.30, 0.76),  # Atlético MG (CA Mineiro)
    1768: (1.22, 0.82),  # Sport Club Internacional
    1767: (1.18, 0.84),  # Grêmio FBPA
    1764: (1.12, 0.88),  # Corinthians
    1762: (1.12, 0.88),  # São Paulo FC
    1769: (1.08, 0.90),  # Cruzeiro
    1770: (1.18, 0.84),  # Botafogo FR
    6685: (1.02, 0.96),  # Santos FC
    1780: (1.05, 0.93),  # CR Vasco da Gama
    1777: (1.05, 0.94),  # EC Bahia
    1782: (0.95, 1.04),  # EC Vitória
    1771: (1.00, 0.98),  # Atletico Paranaense
    4286: (1.12, 0.88),  # RB Bragantino
    4364: (0.93, 1.03),  # Mirassol FC
    1772: (0.88, 1.10),  # Chapecoense AF
    1773: (0.98, 1.00),  # Coritiba
    1774: (0.95, 1.02),  # Fortaleza EC
    1775: (0.93, 1.03),  # Goiás EC
    1776: (0.92, 1.04),  # Cuiabá EC
    1779: (0.90, 1.06),  # América MG
    1783: (0.92, 1.04),  # Ceará SC

    # ── Eredivisie ────────────────────────────────────────────────────────────
    674: (1.52, 0.68),   # Ajax
    683: (1.42, 0.72),   # PSV Eindhoven
    682: (1.30, 0.80),   # Feyenoord
    688: (1.05, 0.93),   # AZ Alkmaar
    675: (1.00, 0.97),   # Utrecht
    691: (0.98, 0.99),   # Vitesse
    693: (0.95, 1.02),   # Twente

    # ── Primeira Liga ─────────────────────────────────────────────────────────
    503: (1.52, 0.68),   # SL Benfica
    498: (1.45, 0.72),   # FC Porto
    5601: (1.35, 0.78),  # Sporting CP
    504: (1.05, 0.93),   # SC Braga
    228: (0.98, 1.00),   # Vitória SC
}

# ── Secondary lookup: normalised name → (attack, defence) ────────────────────
# Used when the team ID is not in the primary table (e.g. newly promoted sides).
RATINGS_BY_NAME: dict[str, tuple[float, float]] = {
    # National teams
    "argentina":        (1.60, 0.68),
    "brazil":           (1.55, 0.70),
    "france":           (1.50, 0.72),
    "spain":            (1.45, 0.72),
    "england":          (1.38, 0.76),
    "germany":          (1.42, 0.74),
    "portugal":         (1.40, 0.75),
    "netherlands":      (1.32, 0.80),
    "belgium":          (1.28, 0.80),
    "italy":            (1.22, 0.82),
    "uruguay":          (1.28, 0.78),
    "colombia":         (1.25, 0.82),
    "mexico":           (1.18, 0.85),
    "croatia":          (1.20, 0.82),
    "switzerland":      (1.18, 0.83),
    "denmark":          (1.15, 0.85),
    "poland":           (1.12, 0.87),
    "serbia":           (1.15, 0.85),
    "morocco":          (1.15, 0.84),
    "senegal":          (1.12, 0.87),
    "japan":            (1.10, 0.87),
    "ecuador":          (1.08, 0.90),
    "usa":              (1.10, 0.88),
    "united states":    (1.10, 0.88),
    "canada":           (1.05, 0.92),
    "australia":        (1.02, 0.93),
    "south korea":      (1.05, 0.92),
    "ghana":            (1.00, 0.95),
    "cameroon":         (1.02, 0.93),
    "nigeria":          (1.08, 0.90),
    "saudi arabia":     (0.98, 0.97),
    "iran":             (0.97, 0.98),
    "qatar":            (0.95, 0.99),
    # Club teams (normalised lower)
    "real madrid":              (1.55, 0.68),
    "barcelona":                (1.45, 0.72),
    "manchester city":          (1.48, 0.70),
    "liverpool":                (1.40, 0.76),
    "arsenal":                  (1.32, 0.78),
    "paris saint-germain":      (1.62, 0.62),
    "psg":                      (1.62, 0.62),
    "bayern münchen":           (1.58, 0.65),
    "bayern munich":            (1.58, 0.65),
    "inter":                    (1.38, 0.72),
    "napoli":                   (1.32, 0.76),
    "atletico madrid":          (1.15, 0.72),
    "atlético madrid":          (1.15, 0.72),
    "bayer leverkusen":         (1.35, 0.72),
    "borussia dortmund":        (1.30, 0.82),
    "rb leipzig":               (1.22, 0.80),
    "ajax":                     (1.52, 0.68),
    "psv eindhoven":            (1.42, 0.72),
    "psv":                      (1.42, 0.72),
    "feyenoord":                (1.30, 0.80),
    "benfica":                  (1.52, 0.68),
    "sl benfica":               (1.52, 0.68),
    "porto":                    (1.45, 0.72),
    "fc porto":                 (1.45, 0.72),
    "sporting cp":              (1.35, 0.78),
    "flamengo":                 (1.38, 0.72),
    "cr flamengo":              (1.38, 0.72),
    "palmeiras":                (1.35, 0.72),
    "atletico mg":              (1.30, 0.76),
    "ca mineiro":               (1.30, 0.76),
    "atletico mineiro":         (1.30, 0.76),
    "botafogo":                 (1.18, 0.84),
    "botafogo fr":              (1.18, 0.84),
    "fluminense":               (1.28, 0.78),
    "fluminense fc":            (1.28, 0.78),
    "gremio":                   (1.18, 0.84),
    "grêmio fbpa":              (1.18, 0.84),
    "corinthians":              (1.12, 0.88),
    "são paulo":                (1.12, 0.88),
    "sao paulo":                (1.12, 0.88),
    "internacional":            (1.22, 0.82),
    "cruzeiro":                 (1.08, 0.90),
    "rb bragantino":            (1.12, 0.88),
    "santos":                   (1.02, 0.96),
    "santos fc":                (1.02, 0.96),
    "vasco da gama":            (1.05, 0.93),
    "cr vasco da gama":         (1.05, 0.93),
    "ec bahia":                 (1.05, 0.94),
    "ec vitória":               (0.95, 1.04),
    "vitoria":                  (0.95, 1.04),
    "chapecoense":              (0.88, 1.10),
    "chapecoense af":           (0.88, 1.10),
    "mirassol":                 (0.93, 1.03),
    "mirassol fc":              (0.93, 1.03),
}


def get_team_strength(team_id: int | None, team_name: str = "") -> tuple[float, float]:
    """
    Return (attack, defence) for a team.

    Lookup order:
      1. team_id   → RATINGS_BY_ID
      2. team_name → RATINGS_BY_NAME  (normalised to lower-case)
      3. Deterministic noise around (1.0, 1.0) for truly unknown sides.
    """
    if team_id and team_id in RATINGS_BY_ID:
        return RATINGS_BY_ID[team_id]

    key = team_name.lower().strip()
    if key in RATINGS_BY_NAME:
        return RATINGS_BY_NAME[key]

    # Partial name match (handles "FC Barcelona" → "barcelona")
    for rated_name, rating in RATINGS_BY_NAME.items():
        if rated_name in key or key in rated_name:
            return rating

    # Unknown team — use deterministic noise so the same team always gets
    # the same slight deviation from average (avoids wild swings).
    seed = f"unknown:{team_id}:{key}"
    rng = random.Random(seed)
    att = round(rng.uniform(0.90, 1.12), 3)
    dff = round(rng.uniform(0.90, 1.10), 3)
    return att, dff
