import logging
from src.formatting import clean_feature_name, clean_ohe_value

logger = logging.getLogger(__name__)


class HypothesisGenerator:
    def __init__(self):
        pass

    def generate(self, insight: dict) -> str:
        itype = insight.get("type")

        if itype == "correlation":
            direction   = insight.get("direction")
            col_a_clean = clean_feature_name(insight.get("col_a", "A"))
            col_b_clean = clean_feature_name(insight.get("col_b", "B"))
            r           = abs(insight.get("r_value", 0))
            strength    = "very strong" if r > 0.75 else ("strong" if r > 0.55 else "moderate")
            if direction == "negative":
                return (
                    f"The {strength} **inverse relationship** between **{col_a_clean}** and **{col_b_clean}** "
                    f"suggests a substitution or resource-constraint effect — when one rises, systemic pressure "
                    f"reduces the other. Consider running pricing experiments or resource-allocation audits "
                    f"to identify the causal bottleneck."
                )
            else:
                return (
                    f"The {strength} **positive co-movement** between **{col_a_clean}** and **{col_b_clean}** "
                    f"points to mutual reinforcement or a shared external driver. "
                    f"Strategies that amplify **{col_a_clean}** are likely to organically boost **{col_b_clean}** "
                    f"as well — consider cross-promotions or bundled initiatives to exploit this leverage point."
                )

        elif itype == "cluster":
            c_int    = insight.get("cluster_id", 0)
            features = insight.get("defining_features", {})
            f_str    = ", ".join([clean_feature_name(f) for f in features.keys()])
            size     = insight.get("size_pct", 0)
            return (
                f"**Segment {c_int + 1}** ({size:.1f}% of data) represents a distinctly identifiable "
                f"sub-population defined by unusual patterns in **{f_str}**. "
                f"Designing tailored interventions — dedicated campaigns, custom pricing, or specialised "
                f"workflows — for this cohort is likely to yield disproportionately high ROI."
            )

        elif itype == "association":
            ant_clean = clean_ohe_value(insight.get("antecedent", "A"))
            con_clean = clean_ohe_value(insight.get("consequent", "B"))
            lift      = insight.get("lift", 1.0)
            return (
                f"The **{lift:.1f}× lift** between {ant_clean} and {con_clean} "
                f"indicates a non-random, systematic dependency — these categories consistently co-occur "
                f"far beyond what chance would predict. "
                f"This likely reflects a genuine behavioural pattern, sequential process, or operational coupling. "
                f"Bundle these offerings, create automated recommendations, or investigate whether a hidden "
                f"third variable is driving both."
            )

        elif itype == "time_lag":
            cause_clean  = clean_feature_name(insight.get("cause", "X"))
            effect_clean = clean_feature_name(insight.get("effect", "Y"))
            lag          = insight.get("lag_days", 1)
            return (
                f"**{cause_clean}** functions as a reliable **early-warning signal** for **{effect_clean}**, "
                f"with a predictable {lag}-day delay. This lag creates a practical operational window: "
                f"monitoring **{cause_clean}** today allows proactive preparation — staffing, inventory, "
                f"or pricing adjustments — before **{effect_clean}** shifts become visible."
            )

        elif itype == "anomaly":
            col_clean = clean_feature_name(insight.get("column", insight.get("col", "this field")))
            count     = insight.get("outlier_count", 1)
            return (
                f"The **{count} extreme value(s)** detected in **{col_clean}** lie far outside the "
                f"normal statistical envelope. These anomalies may reflect genuine exceptional events "
                f"(e.g., promotions, errors, seasonal spikes), data quality issues, or emergent outlier "
                f"behaviour that warrants immediate investigation. Anomalies often contain the most "
                f"actionable intelligence in a dataset."
            )

        elif itype == "distribution":
            col_clean = clean_feature_name(insight.get("column", "this field"))
            top_val   = insight.get("top_value", "the dominant category")
            top_pct   = insight.get("top_pct", 0)
            return (
                f"The heavy concentration of **{col_clean}** around **{top_val}** ({top_pct:.1f}% of records) "
                f"signals a structural bias in the data. Models trained on this distribution without balancing "
                f"techniques will systematically over-predict **{top_val}** outcomes. "
                f"Explore whether this dominance is organic demand, data collection bias, or a market reality "
                f"that should be factored into strategic planning."
            )

        elif itype == "numeric_insight":
            col_clean = clean_feature_name(insight.get("column", "this field"))
            skew      = insight.get("skew", None)
            if skew is not None:
                direction = "upper-end outliers" if skew > 0 else "lower-end outliers"
                return (
                    f"The skewed distribution of **{col_clean}** means the arithmetic mean is being "
                    f"pulled toward {direction}, creating a misleading average. "
                    f"Decision-makers relying on the mean may significantly over- or under-estimate "
                    f"typical performance. Switching to median-based KPIs or log-scaling this variable "
                    f"will produce more reliable baselines."
                )
            cv = insight.get("cv", None)
            if cv is not None:
                return (
                    f"The high variability in **{col_clean}** (CV = {cv:.1f}%) suggests this metric "
                    f"is driven by multiple competing sub-populations or highly context-dependent conditions. "
                    f"Aggregating it into a single average without segmentation will mask critical "
                    f"differences between groups. Segment-level analysis is strongly recommended."
                )

        return "This pattern requires additional domain context for a full hypothesis."

    def generate_all(self, insights: list) -> list:
        logger.info("Generating hypotheses...")
        for insight in insights:
            insight["hypothesis"] = self.generate(insight)
        return insights


def run_pipeline(filepath: str):
    pass
