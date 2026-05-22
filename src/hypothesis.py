import logging
from src.formatting import clean_feature_name, clean_ohe_value

logger = logging.getLogger(__name__)

class HypothesisGenerator:
    def __init__(self):
        pass

    def generate(self, insight: dict) -> str:
        itype = insight.get("type")
        
        if itype == "correlation":
            direction = insight.get("direction")
            col_a_clean = clean_feature_name(insight.get("col_a", "A"))
            col_b_clean = clean_feature_name(insight.get("col_b", "B"))
            if direction == "negative":
                return f"The inverse relationship between **{col_a_clean}** and **{col_b_clean}** suggests a substitution or constraint effect. When **{col_b_clean}** is elevated, it puts downward pressure on **{col_a_clean}**. Consider running target campaigns or pricing adjustments to optimize performance when either is at a seasonal low."
            else:
                return f"The strong positive correlation between **{col_a_clean}** and **{col_b_clean}** indicates mutual reinforcement or a common external driver. Strategies that amplify **{col_a_clean}** will likely yield organic growth in **{col_b_clean}** as well. Explore cross-promotions to leverage this connection."

        elif itype == "cluster":
            c_int = insight.get("cluster_id", 0)
            features = insight.get("defining_features", {})
            f_str = ", ".join([clean_feature_name(f) for f in features.keys()])
            return f"**Cluster {c_int}** represents a highly specialized customer profile or operational state, uniquely defined by variations in **{f_str}**. Designing specialized outreach, exclusive offerings, or custom operations tailored to this specific profile holds high strategic value."

        elif itype == "association":
            ant_clean = clean_ohe_value(insight.get("antecedent", "A"))
            con_clean = clean_ohe_value(insight.get("consequent", "B"))
            return f"There is an incredibly strong affinity between {ant_clean} and {con_clean}. This likely represents a sequential consumer choice or a natural bundle dependency. We suggest positioning these elements together, pre-bundling them, or setting up automated recommendations."

        elif itype == "time_lag":
            cause_clean = clean_feature_name(insight.get("cause", "X"))
            effect_clean = clean_feature_name(insight.get("effect", "Y"))
            lag = insight.get("lag_days", 1)
            return f"Changes in **{cause_clean}** consistently foreshadow adjustments in **{effect_clean}** after a **{lag}-day buffer**. This delay offers an excellent operational window: you can monitor **{cause_clean}** today to accurately predict and proactively staff or stock for **{effect_clean}** tomorrow."
            
        return "Needs more context for hypothesis generation."

    def generate_all(self, insights: list) -> list:
        logger.info("Generating hypotheses...")
        for insight in insights:
            insight["hypothesis"] = self.generate(insight)
        return insights

def run_pipeline(filepath: str):
    pass

