import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.formatting import clean_feature_name

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
            
            drivers = []
            for col, imp in feat_imp.head(5).items():
                drivers.append({
                    "feature": clean_feature_name(col),
                    "raw_col": col,
                    "importance": float(imp * 100)
                })
            
            X_mod = X.copy()
            X_mod[change_col] = X_mod[change_col] * (1 + pct_change / 100.0)
            
            pred_orig = model.predict(X).mean()
            pred_mod = model.predict(X_mod).mean()
            
            delta_pct = ((pred_mod - pred_orig) / pred_orig * 100) if pred_orig != 0 else 0
            clean_t = clean_feature_name(target_col)
            clean_c = clean_feature_name(change_col)
            delta_val = pred_mod - pred_orig
            direction = "Increase" if delta_pct > 0 else "Decrease"
            
            # Formulate strategic business recommendation
            if delta_pct > 2.0:
                recommendation = f"💡 **Strategic Recommendation:** Increasing **{clean_c}** is predicted to significantly boost **{clean_t}**. If this aligns with your business goals, consider allocating additional resource or campaign budget to **{clean_c}** to capitalize on this high-leverage driver."
            elif delta_pct < -2.0:
                recommendation = f"⚠️ **Risk Advisory:** Raising **{clean_c}** is predicted to suppress **{clean_t}** by {abs(delta_pct):.1f}%. We recommend exercising caution and potentially reducing exposure or optimizing this interaction to prevent unintended negative side-effects."
            else:
                recommendation = f"ℹ️ **Strategic Insight:** Modifying **{clean_c}** has a minor impact (about {delta_pct:.2f}%) on **{clean_t}**. This suggests **{clean_t}** is highly inelastic with respect to **{clean_c}**. We recommend prioritizing resources on stronger driver variables."
                
            narrative = (
                f"**AI Prediction Summary:** If you alter **{clean_c}** by **{pct_change:+.1f}%**, "
                f"expect **{clean_t}** to **{direction.lower()}** by **{abs(delta_pct):.1f}%** on average.\n\n"
                f"*(Baseline average moves from {pred_orig:.2f} --> {pred_mod:.2f} units)*"
            )
            
            return {
                "change_col": change_col,
                "target_col": target_col,
                "pct_change_input": float(pct_change),
                "predicted_original_mean": float(pred_orig),
                "predicted_modified_mean": float(pred_mod),
                "delta_pct": float(delta_pct),
                "drivers": drivers,
                "narrative": narrative,
                "recommendation": recommendation
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
