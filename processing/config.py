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
TRAIN_END = "2018-12-31"
VAL_END = "2022-12-31"
MIN_DATE = "2005-01-03"

# Fundamental files: internal name → (relative path from RAW_ECO, column name in long format)
FUNDAMENTAL_FILES = {
    "roa": ("trimestral/ROA.csv", "roa"),
    "roe": ("trimestral/ROE.csv", "roe"),
    "margem_bruta": ("trimestral/margembruta.csv", "margem_bruta"),
    "divida_bruta_ativo": ("trimestral/dividabruta_ativo.csv", "divida_bruta_ativo"),
    "divida_liq_pl": ("trimestral/dividaliq_pl.csv", "divida_liq_pl"),
    "pvpa": ("diario/preco_valor_patrimonial.csv", "pvpa"),
    "ev_ebitda": ("diario/ev_ebitda.csv", "ev_ebitda"),
    "preco_lucro": ("diario/preco_lucro.csv", "preco_lucro"),
    "volume": ("diario/volume.csv", "volume"),
}

BLOOMBERG_PATH = RAW / "bloomberg_indices_values.xlsx"
SECTOR_PATH = RAW / "setor_ibovespa.xlsx"

# Composite indicator raw sources
FCF_PATH = RAW_ECO / "trimestral/fluxodecaixalivre.csv"
DIVIDA_TOTAL_PATH = RAW_ECO / "trimestral/dividatotalbruta.csv"
MKTCAP_PATH = RAW_ECO / "diario/valordemercado.csv"
