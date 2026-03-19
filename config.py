import numpy as np
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

HORIZON = 8
SEASON_LEN = 4  # Квартальная сезонность
N_SERIES = 200

DATASET_FOLDER = Path(__file__).resolve().parent / "datasets"
