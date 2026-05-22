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
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if numeric_df.shape[1] < 2:
            return []
            
        # ── 1. Filter out useless simulation targets (IDs, Episodes, Dates) ──
        bad_keywords = ['id', 'episode', 'season', 'year', 'month', 'day', 'week', 'index', 'zip']
        valid_cols = []
        for col in numeric_df.columns:
            lower_col = col.lower()
            # Ignore purely categorical/ordinal columns with very few unique values
            if not any(bw in lower_col for bw in bad_keywords) and numeric_df[col].nunique() > 2:
                valid_cols.append(col)
                
        if len(valid_cols) < 2:
            # Fallback if the filter is too aggressive
            valid_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() > 1]
            if len(valid_cols) < 2:
                return []
                
        # ── 2. Prioritize Business KPIs as Targets ──
        target_kpis = ['amount', 'val', 'sales', 'revenue', 'profit', 'equity', 'price', 'spend', 'cost', 'score', 'units', 'traffic']
        potential_targets = []
        for col in valid_cols:
            if any(tk in col.lower() for tk in target_kpis):
                potential_targets.append(col)
                
        if not potential_targets:
            potential_targets = valid_cols
            
        # ── 3. Find strong relationships to simulate ──
        corr = numeric_df[valid_cols].corr().abs()
        results = []
        seen_pairs = set()
        
        # Pass A: Look for scenarios that actually shift the outcome by > 0.5%
        for target in potential_targets:
            related_cols = corr[target].sort_values(ascending=False).index.tolist()
            for change in related_cols:
                if target == change:
                    continue
                pair = frozenset([target, change])
                if pair in seen_pairs:
                    continue
                    
                # Try a +20% intervention
                res = self.simulate(df, target, change, 20.0)
                if "error" not in res:
                    if abs(res["delta_pct"]) > 0.5:
                        results.append(res)
                        seen_pairs.add(pair)
                        
                if len(results) >= 3:
                    return results
                    
        # Pass B: Fallback to any valid pairs if we couldn't find impactful ones
        if len(results) < 3:
            for target in potential_targets:
                related_cols = corr[target].sort_values(ascending=False).index.tolist()
                for change in related_cols:
                    if target == change:
                        continue
                    pair = frozenset([target, change])
                    if pair not in seen_pairs:
                        res = self.simulate(df, target, change, 10.0)
                        if "error" not in res:
                            results.append(res)
                            seen_pairs.add(pair)
                        if len(results) >= 3:
                            return results
                            
        return results

def run_pipeline(filepath: str):
    pass
