from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd


sys.path.append(str(Path(__file__).parent.parent))

from scripts.fetch_data import FetchData



def load_dataset(path: Path, refresh: bool = False) -> pd.DataFrame:
    """Load the local CSV, fetching UCI dataset 144 only when necessary."""
    if path.exists() and not refresh:
        return pd.read_csv(path)

    fetcher = FetchData()
    dataset = fetcher.fetch_data(id=144)
    frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame

