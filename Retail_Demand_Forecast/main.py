from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd


sys.path.append(str(Path(__file__).parent.parent))


data = pd.read_csv("Retail_Demand_Forecast/data/retail_timeseries_2yr.csv")
print(data.head())

#label the data
target = data['Total_Qty_CTN']
print(target.head())