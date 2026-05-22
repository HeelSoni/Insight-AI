import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from src.formatting import clean_feature_name, clean_ohe_value


class QuestionGenerator:
    def __init__(self):
        pass

    def generate(self, insight: dict) -> list:
        itype = insight.get("type")
        questions = []

        if itype == "correlation":
            col_a_clean = clean_feature_name(insight.get("col_a", "A"))
            col_b_clean = clean_feature_name(insight.get("col_b", "B"))
            questions = [
                f"What underlying operational or market drivers best explain the link between **{col_a_clean}** and **{col_b_clean}**?",
                f"Is this relationship consistent year-round, or does it intensify during specific periods?",
                f"Are there any confounding variables simultaneously influencing both **{col_a_clean}** and **{col_b_clean}**?",
            ]

        elif itype == "cluster":
            c_int = insight.get("cluster_id", 0) + 1
            features = insight.get("defining_features", {})
            f_str = ", ".join([clean_feature_name(f) for f in list(features.keys())[:2]])
            questions = [
                f"What specific behaviours or conditions define Segment {c_int} compared to other groups?",
                f"What trigger events or seasonal patterns explain the formation of this segment?",
                f"How can we design a targeted intervention specifically optimised for Segment {c_int}?",
            ]

        elif itype == "association":
            ant_clean = clean_ohe_value(insight.get("antecedent", "A")).replace("**", "")
            con_clean = clean_ohe_value(insight.get("consequent", "B")).replace("**", "")
            lift      = insight.get("lift", 1.0)
            questions = [
                f"What process or behavioural logic explains why **{ant_clean}** so reliably co-occurs with **{con_clean}**?",
                f"Does this {lift:.1f}× lift hold consistently across all customer segments and time periods?",
                f"How can we proactively use **{ant_clean}** as a trigger to maximise **{con_clean}** outcomes?",
            ]

        elif itype == "time_lag":
            cause_clean  = clean_feature_name(insight.get("cause", "X"))
            effect_clean = clean_feature_name(insight.get("effect", "Y"))
            lag          = insight.get("lag_days", 1)
            questions = [
                f"What process or physical bottleneck causes the **{lag}-day delay** between **{cause_clean}** and **{effect_clean}**?",
                f"Does this lag shorten or lengthen during peak seasons or high-demand periods?",
                f"Can we set up automated alerts on **{cause_clean}** to trigger proactive **{effect_clean}** preparation?",
            ]

        elif itype == "anomaly":
            col_clean = clean_feature_name(insight.get("column", insight.get("col", "this metric")))
            count     = insight.get("outlier_count", 1)
            questions = [
                f"What event or condition caused the **{count} extreme value(s)** in **{col_clean}**?",
                f"Are these outliers genuine exceptional events, or are they data quality / recording errors?",
                f"Should these anomalies be excluded from modelling, or do they represent the most important edge cases to understand?",
            ]

        elif itype == "distribution":
            col_clean = clean_feature_name(insight.get("column", "this field"))
            top_val   = insight.get("top_value", "the dominant category")
            n_unique  = insight.get("n_unique", 2)
            questions = [
                f"Is the dominance of **{top_val}** in **{col_clean}** a genuine market reality or a data collection artefact?",
                f"How does model performance change if we apply class balancing across the {n_unique} categories?",
                f"Which minority categories in **{col_clean}** represent the highest-growth or highest-risk opportunities?",
            ]

        elif itype == "numeric_insight":
            col_clean = clean_feature_name(insight.get("column", "this metric"))
            questions = [
                f"What specific sub-groups or conditions are driving the unusual distribution of **{col_clean}**?",
                f"Should we replace the mean KPI for **{col_clean}** with a median or percentile-based metric?",
                f"Does the variability in **{col_clean}** correlate with any categorical segmentation variable?",
            ]

        return questions

    def generate_all(self, insights: list, df: pd.DataFrame = None) -> list:
        logger.info("Generating strategic questions...")
        for insight in insights:
            insight["questions"] = self.generate(insight)
        return insights


def run_pipeline(filepath: str):
    pass
