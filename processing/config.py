"""Shared constants and paths for the data pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
RAW_ECO = RAW / "economatica"

CLEANED = ROOT / "cleaned"
FEATURES = ROOT / "features"
PARQUETS = ROOT / "parquets"

# Ensure output directories exist
for d in (CLEANED, FEATURES, PARQUETS):
    d.mkdir(exist_ok=True)

# Temporal split boundaries
TRAIN_END = "2017-12-31"
VAL_END   = "2021-12-31"
MIN_DATE = "2005-01-03"

# Fundamental files: internal name → (relative path from RAW_ECO, column name in long format)
FUNDAMENTAL_FILES = {
    "roa": ("trimestral/ROA.csv", "roa"),
    "roe": ("trimestral/ROE.csv", "roe"),
    "margem_bruta": ("trimestral/margembruta.csv", "margem_bruta"),
    "divida_bruta_ativo": ("trimestral/dividabruta_ativo.csv", "divida_bruta_ativo"),
    "divida_liq_pl": ("trimestral/dividaliq_pl.csv", "divida_liq_pl"),
    "pvpa": ("diario/preco_valor_patrimonial.csv", "pvpa"),
    "preco_lucro": ("diario/preco_lucro.csv", "preco_lucro"),
    "volume": ("diario/volume.csv", "volume"),
}

BLOOMBERG_PATH = RAW / "bloomberg_indices_values.xlsx"
SECTOR_PATH = RAW / "setor_ibovespa.xlsx"

# Forward-fill limit for quarterly fundamentals (trading days).
# 65 instead of 63 to cover the off-by-one where Brazilian quarter-end
# report dates (e.g. Sep 30) fall exactly 64 trading days after the
# prior quarter-end (Jun 30), leaving a 1-day gap at the boundary.
FFILL_LIMIT = 65

# Liquidity filter applied in 01_clean_prices.py
# Remove a ticker if >= VOL_MIN_FRAC_ABOVE of its trading days have volume >= VOL_THRESHOLD_K
VOL_THRESHOLD_K    = 1_000   # R$1M/day (values in CSV are BRL thousands)
VOL_MIN_FRAC_ABOVE = 0.90    # keep only if >= 90% of days meet threshold

# Composite indicator raw sources
FCF_PATH = RAW_ECO / "trimestral/fluxodecaixalivre.csv"
DIVIDA_TOTAL_PATH = RAW_ECO / "trimestral/dividatotalbruta.csv"
MKTCAP_PATH = RAW_ECO / "diario/valordemercado.csv"
