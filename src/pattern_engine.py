import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
import logging
from src.formatting import clean_feature_name, clean_ohe_value


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternEngine:
    def __init__(self):
        pass

    def _find_correlations(self, df: pd.DataFrame) -> list:
        insights = []
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return insights
            
        pearson_corr = numeric_df.corr(method='pearson')
        spearman_corr = numeric_df.corr(method='spearman')
        
        cols = pearson_corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_a = cols[i]
                col_b = cols[j]
                r_pearson = pearson_corr.iloc[i, j]
                r_spearman = spearman_corr.iloc[i, j]
                
                if pd.notna(r_pearson) and abs(r_pearson) > 0.4:
                    direction = "positive" if r_pearson > 0 else "negative"
                    col_a_clean = clean_feature_name(col_a)
                    col_b_clean = clean_feature_name(col_b)
                    desc = f"📈 **{col_a_clean}** and **{col_b_clean}** move in the **{direction}** direction. When one rises, the other tends to {'increase' if direction == 'positive' else 'decrease'} (correlation score of {r_pearson:.2f})."
                    
                    if pd.notna(r_spearman) and abs(r_pearson - r_spearman) > 0.2:
                        desc += " The relationship appears to have non-linear properties, indicating complex scaling."
                        
                    insights.append({
                        "type": "correlation",
                        "col_a": col_a,
                        "col_b": col_b,
                        "r_value": float(r_pearson),
                        "direction": direction,
                        "description": desc
                    })
        return insights

    def _find_clusters(self, df: pd.DataFrame) -> list:
        insights = []
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2 or len(numeric_df) < 10:
            return insights

        numeric_df = numeric_df.fillna(numeric_df.median())

        best_k = 3
        best_score = -1
        for k in range(2, 7):
            if k >= len(numeric_df):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(numeric_df)
            score = silhouette_score(numeric_df, labels)
            if score > best_score:
                best_score = score
                best_k = k

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(numeric_df)
        df_clustered = numeric_df.copy()
        df_clustered['cluster'] = labels
        
        for c in range(best_k):
            cluster_data = df_clustered[df_clustered['cluster'] == c]
            size_pct = len(cluster_data) / len(df_clustered) * 100
            
            means = cluster_data.drop(columns=['cluster']).mean()
            overall_means = numeric_df.mean()
            diffs = abs(means - overall_means) / (overall_means.replace(0, 1e-9))
            top_features = diffs.nlargest(3).index.tolist()
            
            feature_dict = {f: float(means[f]) for f in top_features}
            clean_features = [clean_feature_name(f) for f in top_features]
            desc = f"👥 **Cluster {c}** (representing **{size_pct:.1f}%** of the data) is highly distinctive. This group is characterized by distinct values in **{', '.join(clean_features)}**."
            
            insights.append({
                "type": "cluster",
                "cluster_id": c,
                "size_pct": float(size_pct),
                "defining_features": feature_dict,
                "silhouette_score": float(best_score),
                "description": desc
            })
            
        return insights

    def _find_associations(self, df: pd.DataFrame) -> list:
        insights = []
        if not MLXTEND_AVAILABLE:
            logger.warning("mlxtend not available, skipping association rules")
            return insights

        cat_df = df.select_dtypes(exclude=[np.number, 'datetime64[ns]'])
        if cat_df.shape[1] < 2:
            return insights

        ohe_df = pd.get_dummies(cat_df)
        if ohe_df.empty or ohe_df.shape[1] > 50:
            return insights

        try:
            frequent_itemsets = apriori(ohe_df, min_support=0.1, use_colnames=True)
            if frequent_itemsets.empty:
                return insights
                
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            if rules.empty:
                return insights
                
            rules = rules[rules['confidence'] >= 0.6]
            rules = rules.sort_values('lift', ascending=False).head(10)
            
            for _, row in rules.iterrows():
                ant = list(row['antecedents'])[0] if row['antecedents'] else ""
                con = list(row['consequents'])[0] if row['consequents'] else ""
                
                ant_clean = clean_ohe_value(str(ant))
                con_clean = clean_ohe_value(str(con))
                desc = f"🛍️ When {ant_clean}, there is a **{row['confidence']*100:.0f}% probability** that {con_clean} (this occurs {row['lift']:.2f}x more frequently than under normal random conditions)."
                insights.append({
                    "type": "association",
                    "antecedent": str(ant),
                    "consequent": str(con),
                    "support": float(row['support']),
                    "confidence": float(row['confidence']),
                    "lift": float(row['lift']),
                    "description": desc
                })
        except Exception as e:
            logger.error(f"Error in association rules: {e}")
            
        return insights

    def _find_time_lags(self, df: pd.DataFrame) -> list:
        insights = []
        date_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
                
        if not date_col:
            return insights

        df_sorted = df.sort_values(by=date_col)
        numeric_df = df_sorted.select_dtypes(include=[np.number])
        
        cols = numeric_df.columns
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i == j:
                    continue
                col_a = cols[i]
                col_b = cols[j]
                
                best_lag = 0
                best_corr = 0
                
                for lag in [1, 2, 3, 5, 7]:
                    shifted_a = numeric_df[col_a].shift(lag)
                    corr = shifted_a.corr(numeric_df[col_b])
                    if pd.notna(corr) and abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                        
                if best_lag > 0 and abs(best_corr) > 0.35:
                    col_a_clean = clean_feature_name(col_a)
                    col_b_clean = clean_feature_name(col_b)
                    desc = f"⏳ **{col_a_clean}** acts as a leading indicator for **{col_b_clean}** with a delayed response of **{best_lag} days** (correlation of {best_corr:.2f})."
                    insights.append({
                        "type": "time_lag",
                        "cause": col_a,
                        "effect": col_b,
                        "lag_days": best_lag,
                        "r_value": float(best_corr),
                        "description": desc
                    })
        return insights


    def discover(self, df: pd.DataFrame) -> list:
        logger.info("Discovering patterns...")
        insights = []
        insights.extend(self._find_correlations(df))
        insights.extend(self._find_clusters(df))
        insights.extend(self._find_associations(df))
        insights.extend(self._find_time_lags(df))
        return insights

def run_pipeline(filepath: str):
    pass
