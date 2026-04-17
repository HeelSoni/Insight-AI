import pandas as pd
import numpy as np
import sys
import os

# Ensure the correct path
sys.path.append(os.path.abspath('C:\\Users\\Heel\\OneDrive\\Desktop\\Insight AI\\insight_ai'))
from src.simulator import CounterfactualSimulator

try:
    df = pd.DataFrame({
        'A': np.random.rand(10),
        'B': np.random.rand(10),
        'C': np.random.rand(10)
    })

    sim = CounterfactualSimulator()
    print("Running simulate_top_pairs...")
    res = sim.simulate_top_pairs(df)
    print("Result:", res)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
