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

    # ─────────────────────────────────────────────
    # 1. CORRELATIONS  (numeric ↔ numeric)
    # ─────────────────────────────────────────────
    def _find_correlations(self, df: pd.DataFrame) -> list:
        insights = []
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return insights

        pearson_corr  = numeric_df.corr(method='pearson')
        spearman_corr = numeric_df.corr(method='spearman')

        cols = pearson_corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_a = cols[i]
                col_b = cols[j]
                r_pearson  = pearson_corr.iloc[i, j]
                r_spearman = spearman_corr.iloc[i, j]

                if pd.notna(r_pearson) and abs(r_pearson) > 0.3:
                    direction   = "positive" if r_pearson > 0 else "negative"
                    col_a_clean = clean_feature_name(col_a)
                    col_b_clean = clean_feature_name(col_b)
                    strength    = "strong" if abs(r_pearson) > 0.65 else "moderate"
                    desc = (
                        f"📈 **{col_a_clean}** and **{col_b_clean}** have a **{strength} {direction} "
                        f"correlation** (r = {r_pearson:.2f}). When one rises, the other tends to "
                        f"{'increase' if direction == 'positive' else 'decrease'}."
                    )
                    if pd.notna(r_spearman) and abs(r_pearson - r_spearman) > 0.2:
                        desc += " The non-linear Spearman rank divergence suggests a complex, possibly exponential relationship."

                    insights.append({
                        "type":      "correlation",
                        "col_a":     col_a,
                        "col_b":     col_b,
                        "r_value":   float(r_pearson),
                        "direction": direction,
                        "description": desc,
                    })
        return insights

    # ─────────────────────────────────────────────
    # 2. CLUSTERS  (numeric groupings)
    # ─────────────────────────────────────────────
    def _find_clusters(self, df: pd.DataFrame) -> list:
        insights = []
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2 or len(numeric_df) < 10:
            return insights

        numeric_df = numeric_df.fillna(numeric_df.median())

        best_k     = 3
        best_score = -1
        for k in range(2, 7):
            if k >= len(numeric_df):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(numeric_df)
            score  = silhouette_score(numeric_df, labels)
            if score > best_score:
                best_score = score
                best_k     = k

        kmeans      = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels      = kmeans.fit_predict(numeric_df)
        df_clustered = numeric_df.copy()
        df_clustered['cluster'] = labels

        for c in range(best_k):
            cluster_data = df_clustered[df_clustered['cluster'] == c]
            size_pct     = len(cluster_data) / len(df_clustered) * 100

            means         = cluster_data.drop(columns=['cluster']).mean()
            overall_means = numeric_df.mean()
            diffs         = abs(means - overall_means) / (overall_means.replace(0, 1e-9))
            top_features  = diffs.nlargest(3).index.tolist()

            feature_dict  = {f: float(means[f]) for f in top_features}
            clean_features = [clean_feature_name(f) for f in top_features]

            # Build a richer, more specific description per cluster
            feature_details = []
            for f in top_features:
                direction = "above" if means[f] > overall_means[f] else "below"
                pct_diff  = abs(means[f] - overall_means[f]) / (abs(overall_means[f]) + 1e-9) * 100
                feature_details.append(
                    f"**{clean_feature_name(f)}** ({direction} average by {pct_diff:.0f}%)"
                )

            desc = (
                f"👥 **Segment {c + 1}** comprises **{size_pct:.1f}%** of your data. "
                f"This segment is distinctly characterised by: {', '.join(feature_details)}."
            )

            insights.append({
                "type":             "cluster",
                "cluster_id":       c,
                "size_pct":         float(size_pct),
                "defining_features": feature_dict,
                "silhouette_score": float(best_score),
                "description":      desc,
            })

        return insights

    # ─────────────────────────────────────────────
    # 3. ASSOCIATIONS  (categorical → categorical)
    #    Deduplicate strictly by SOURCE COLUMN PAIRS.
    #    High-cardinality columns (IDs/names) are
    #    excluded before OHE so trivial startup→investor
    #    rules never appear. Hard cap: 5 max.
    # ─────────────────────────────────────────────
    def _find_associations(self, df: pd.DataFrame) -> list:
        insights = []
        if not MLXTEND_AVAILABLE:
            logger.warning("mlxtend not available, skipping association rules")
            return insights

        cat_df = df.select_dtypes(exclude=[np.number, 'datetime64[ns]'])
        if cat_df.shape[1] < 2:
            return insights

        n_rows = len(cat_df)

        # ── Drop high-cardinality columns (IDs / startup names / unique combos)
        # Keep only columns whose unique-value count ≤ min(20, 20% of rows).
        # This eliminates "Startup Name", "Sharks Invested" combos, etc.
        max_unique = max(5, int(n_rows * 0.20))
        useful_cat_cols = [
            col for col in cat_df.columns
            if 2 <= cat_df[col].nunique() <= max_unique
        ]

        logger.info(f"Association engine: {len(useful_cat_cols)} low-cardinality cols from {cat_df.shape[1]} total")

        if len(useful_cat_cols) < 2:
            logger.info("Not enough low-cardinality categorical columns for associations.")
            return insights

        cat_filtered = cat_df[useful_cat_cols]

        # ── Build OHE and track ohe_col_name → source_column ─────────────────
        ohe_df = pd.get_dummies(cat_filtered)
        if ohe_df.empty or ohe_df.shape[1] > 200:
            return insights

        # Build reverse map: exact OHE column name → source column name
        ohe_to_source = {}
        for col in useful_cat_cols:
            for val in cat_filtered[col].dropna().unique():
                ohe_col = f"{col}_{val}"
                if ohe_col in ohe_df.columns:
                    ohe_to_source[ohe_col] = col

        try:
            frequent_itemsets = apriori(ohe_df, min_support=0.05, use_colnames=True)
            if frequent_itemsets.empty:
                return insights

            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            if rules.empty:
                return insights

            rules = rules[rules['confidence'] >= 0.4]
            rules = rules.sort_values('lift', ascending=False)

            # ── Strict column-pair deduplication ─────────────────────────────
            # Once we've seen "Industry → Season", skip ALL other variants of it.
            seen_col_pairs = set()

            for _, row in rules.iterrows():
                # Hard cap: stop after 5 association insights
                if len(insights) >= 5:
                    break

                ant_items = sorted([str(x) for x in row['antecedents']])
                con_items = sorted([str(x) for x in row['consequents']])

                # Map OHE column names back to source column names
                ant_src = frozenset(ohe_to_source.get(x, "__unknown__") for x in ant_items)
                con_src = frozenset(ohe_to_source.get(x, "__unknown__") for x in con_items)

                # Skip if ant and con share the same source column (self-reference)
                if ant_src & con_src:
                    continue

                # Skip if we have already seen this column relationship (forward OR reverse)
                if (ant_src, con_src) in seen_col_pairs or (con_src, ant_src) in seen_col_pairs:
                    continue

                seen_col_pairs.add((ant_src, con_src))

                ant_clean = " AND ".join([clean_ohe_value(x) for x in ant_items])
                con_clean = " AND ".join([clean_ohe_value(x) for x in con_items])
                desc = (
                    f"🛍️ When {ant_clean}, there is a "
                    f"**{row['confidence'] * 100:.0f}% probability** that {con_clean} "
                    f"(this occurs **{row['lift']:.2f}×** more frequently than by random chance). "
                    f"Support: {row['support'] * 100:.1f}% of records match this pattern."
                )

                insights.append({
                    "consequent": ", ".join(con_items),
                    "support":    float(row['support']),
                    "confidence": float(row['confidence']),
                    "lift":       float(row['lift']),
                    "description": desc,
                })

        except Exception as e:
            logger.error(f"Error in association rules: {e}")

        return insights

    # ─────────────────────────────────────────────
    # 4. TIME LAGS  (leading / lagging indicators)
    # ─────────────────────────────────────────────
    def _find_time_lags(self, df: pd.DataFrame) -> list:
        insights = []
        date_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break

        if not date_col:
            return insights

        df_sorted  = df.sort_values(by=date_col)
        numeric_df = df_sorted.select_dtypes(include=[np.number])

        cols          = numeric_df.columns
        seen_pairs: set = set()

        for i in range(len(cols)):
            for j in range(len(cols)):
                if i == j:
                    continue
                col_a = cols[i]
                col_b = cols[j]

                pair     = frozenset([col_a, col_b])
                if pair in seen_pairs:
                    continue

                best_lag  = 0
                best_corr = 0

                for lag in [1, 2, 3, 5, 7]:
                    shifted_a = numeric_df[col_a].shift(lag)
                    corr      = shifted_a.corr(numeric_df[col_b])
                    if pd.notna(corr) and abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag  = lag

                if best_lag > 0 and abs(best_corr) > 0.25:
                    seen_pairs.add(pair)
                    col_a_clean = clean_feature_name(col_a)
                    col_b_clean = clean_feature_name(col_b)
                    desc = (
                        f"⏳ **{col_a_clean}** acts as a **leading indicator** for "
                        f"**{col_b_clean}** with a delayed response of **{best_lag} day(s)** "
                        f"(correlation: {best_corr:.2f}). Changes in {col_a_clean} today "
                        f"predict changes in {col_b_clean} in {best_lag} day(s)."
                    )
                    insights.append({
                        "type":     "time_lag",
                        "cause":    col_a,
                        "effect":   col_b,
                        "lag_days": best_lag,
                        "r_value":  float(best_corr),
                        "description": desc,
                    })
        return insights

    # ─────────────────────────────────────────────
    # 5. ANOMALIES  (statistical outliers per col)
    # ─────────────────────────────────────────────
    def _find_anomalies(self, df: pd.DataFrame) -> list:
        insights = []
        numeric_df = df.select_dtypes(include=[np.number])

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 10:
                continue

            mean = series.mean()
            std  = series.std()
            if std == 0:
                continue

            outliers = series[abs(series - mean) > 3 * std]
            if len(outliers) == 0:
                continue

            col_clean   = clean_feature_name(col)
            pct_outlier = len(outliers) / len(series) * 100
            extreme_val = outliers.abs().idxmax()
            extreme_amt = series[extreme_val]

            desc = (
                f"🚨 **{col_clean}** contains **{len(outliers)} statistical outlier(s)** "
                f"({pct_outlier:.1f}% of records) that lie beyond 3 standard deviations "
                f"from the mean ({mean:.2f} ± {std:.2f}). "
                f"The most extreme value is **{extreme_amt:.2f}**, which is "
                f"**{abs(extreme_amt - mean) / std:.1f}σ** from average. "
                f"These anomalies may indicate data entry errors, exceptional events, or genuine outlier behaviour."
            )

            insights.append({
                "type":          "anomaly",
                "column":        col,
                "outlier_count": len(outliers),
                "mean":          float(mean),
                "std":           float(std),
                "description":   desc,
            })

        return insights

    # ─────────────────────────────────────────────
    # 6. DISTRIBUTIONS  (dominant category analysis)
    # ─────────────────────────────────────────────
    def _find_distributions(self, df: pd.DataFrame) -> list:
        insights = []
        cat_df = df.select_dtypes(exclude=[np.number, 'datetime64[ns]'])

        for col in cat_df.columns:
            series   = cat_df[col].dropna()
            n        = len(series)
            if n < 5:
                continue

            vc = series.value_counts()
            n_unique = len(vc)

            # Skip near-unique columns (IDs / names)
            if n_unique > max(10, n * 0.30):
                continue

            top_val   = vc.index[0]
            top_count = vc.iloc[0]
            top_pct   = top_count / n * 100
            col_clean = clean_feature_name(col)

            # Only report if dominant category is genuinely significant
            if top_pct < 20:
                continue

            if n_unique == 2:
                second_val = vc.index[1]
                second_pct = vc.iloc[1] / n * 100
                desc = (
                    f"📊 **{col_clean}** is a binary variable. "
                    f"**{top_val}** dominates at **{top_pct:.1f}%** of records "
                    f"vs **{second_val}** at {second_pct:.1f}%. "
                    f"This strong imbalance can skew any model that uses this field without balancing."
                )
            else:
                top3 = vc.head(3)
                top3_str = ", ".join(
                    [f"**{v}** ({c / n * 100:.1f}%)" for v, c in top3.items()]
                )
                tail_pct = (n - top3.sum()) / n * 100
                desc = (
                    f"📊 **{col_clean}** has **{n_unique} unique categories**. "
                    f"The top 3 are: {top3_str}. "
                    f"The remaining {n_unique - 3} categories account for only "
                    f"**{tail_pct:.1f}%** of the data — meaning this field is highly concentrated."
                )

            insights.append({
                "type":        "distribution",
                "column":      col,
                "top_value":   str(top_val),
                "top_pct":     float(top_pct),
                "n_unique":    n_unique,
                "description": desc,
            })

        return insights

    # ─────────────────────────────────────────────
    # 7. NUMERIC SUMMARIES  (per-column highlights)
    # ─────────────────────────────────────────────
    def _find_numeric_insights(self, df: pd.DataFrame) -> list:
        insights = []
        numeric_df = df.select_dtypes(include=[np.number])

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 5:
                continue

            col_clean = clean_feature_name(col)
            skew      = series.skew()
            q1        = series.quantile(0.25)
            q3        = series.quantile(0.75)
            iqr       = q3 - q1
            median    = series.median()
            mean      = series.mean()
            cv        = series.std() / (abs(mean) + 1e-9) * 100  # coefficient of variation

            if abs(skew) > 1.0:
                skew_dir = "right-skewed (long upper tail — a few very high values pull the mean up)" if skew > 0 \
                           else "left-skewed (long lower tail — a few very low values pull the mean down)"
                desc = (
                    f"📐 **{col_clean}** is strongly **{skew_dir}** (skewness = {skew:.2f}). "
                    f"The median is **{median:.2f}** while the mean is **{mean:.2f}** — "
                    f"a gap of {abs(mean - median):.2f} caused by extreme values. "
                    f"Consider log-transforming this variable before modelling."
                )
                insights.append({
                    "type":        "numeric_insight",
                    "column":      col,
                    "skew":        float(skew),
                    "median":      float(median),
                    "mean":        float(mean),
                    "description": desc,
                })

            elif cv > 50:
                desc = (
                    f"📏 **{col_clean}** shows **high variability** (coefficient of variation = {cv:.1f}%). "
                    f"Values range broadly: the interquartile range spans **{q1:.2f} → {q3:.2f}** (IQR = {iqr:.2f}). "
                    f"This level of dispersion suggests heterogeneous sub-populations or inconsistent conditions."
                )
                insights.append({
                    "type":        "numeric_insight",
                    "column":      col,
                    "cv":          float(cv),
                    "iqr":         float(iqr),
                    "description": desc,
                })

        return insights

    # ─────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────
    def discover(self, df: pd.DataFrame) -> list:
        logger.info("Discovering patterns across all engines...")
        insights = []
        insights.extend(self._find_correlations(df))
        insights.extend(self._find_clusters(df))
        insights.extend(self._find_associations(df))
        insights.extend(self._find_time_lags(df))
        insights.extend(self._find_anomalies(df))
        insights.extend(self._find_distributions(df))
        insights.extend(self._find_numeric_insights(df))
        logger.info(f"Total raw insights discovered: {len(insights)}")
        return insights


def run_pipeline(filepath: str):
    pass
