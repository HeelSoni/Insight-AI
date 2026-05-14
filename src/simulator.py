import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class CounterfactualSimulator:
    def __init__(self):
        pass

    def simulate(self, df: pd.DataFrame, target_col: str, change_col: str, pct_change: float) -> dict:
        try:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            if target_col not in numeric_df.columns or change_col not in numeric_df.columns:
                return {"error": "Columns not numeric or missing"}
            
            X = numeric_df.drop(columns=[target_col])
            y = numeric_df[target_col]
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            top_predictors = feat_imp.head(5).index.tolist()
            
            X_mod = X.copy()
            X_mod[change_col] = X_mod[change_col] * (1 + pct_change / 100.0)
            
            pred_orig = model.predict(X).mean()
            pred_mod = model.predict(X_mod).mean()
            
            delta_pct = ((pred_mod - pred_orig) / pred_orig * 100) if pred_orig != 0 else 0
            clean_t = target_col.replace('_', ' ').title()
            clean_c = change_col.replace('_', ' ').title()
            delta_val = pred_mod - pred_orig
            direction = "Increase" if delta_pct > 0 else "Decrease"
            
            narrative = (
                f"**AI Prediction:** If you raise **{clean_c}** by **{pct_change}%**, "
                f"expect **{clean_t}** to {direction} by **{abs(delta_pct):.1f}%**.\n\n"
                f"*(Average moves from {pred_orig:.1f} → {pred_mod:.1f})*"
            )
            
            return {
                "change_col": change_col,
                "target_col": target_col,
                "pct_change_input": float(pct_change),
                "predicted_original_mean": float(pred_orig),
                "predicted_modified_mean": float(pred_mod),
                "delta_pct": float(delta_pct),
                "top_predictors": top_predictors,
                "narrative": narrative
            }
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {"error": str(e)}

    def simulate_top_pairs(self, df: pd.DataFrame) -> list:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return []
            
        corr = numeric_df.corr().abs()
        corr_array = np.array(corr.values, copy=True)
        np.fill_diagonal(corr_array, 0)
        
        top_pairs = []
        for _ in range(3):
            if corr_array.max() == 0:
                break
            idx = corr_array.argmax()
            r, c = divmod(idx, corr_array.shape[1])
            col_a = corr.index[r]
            col_b = corr.columns[c]
            top_pairs.append((col_a, col_b))
            corr_array[r, c] = 0
            corr_array[c, r] = 0
            
        results = []
        for target, change in top_pairs:
            res = self.simulate(df, target, change, 10.0)
            if "error" not in res:
                results.append(res)
                
        return results

def run_pipeline(filepath: str):
    pass
