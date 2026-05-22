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
                f"What underlying operational or market drivers could explain the link between {col_a_clean} and {col_b_clean}?",
                f"Is this relationship consistent year-round, or does it shift during specific quarters?",
                f"Are there any secondary variables that might be simultaneously influencing both features?"
            ]

        elif itype == "cluster":
            c_int = insight.get("cluster_id", 0)
            questions = [
                f"What specific customer behaviors or system parameters set Cluster {c_int} apart from other cohorts?",
                f"What trigger events or seasonal shifts might explain the formation of Cluster {c_int}?",
                f"How has the user base or sample size inside Cluster {c_int} trended historically?"
            ]

        elif itype == "association":
            ant_clean = clean_ohe_value(insight.get("antecedent", "A"))
            con_clean = clean_ohe_value(insight.get("consequent", "B"))
            # Remove asterisks if they exist for text flow
            ant_text = ant_clean.replace("**", "")
            con_text = con_clean.replace("**", "")
            questions = [
                f"What sequential user journey or system logic explains why {ant_text} strongly triggers {con_text}?",
                f"Does this behavioral link hold universally across all segments, or is it isolated to certain groups?",
                f"What proactive actions can we take to influence or capitalize on {con_text} once we detect {ant_text}?"
            ]

        elif itype == "time_lag":
            cause_clean = clean_feature_name(insight.get("cause", "X"))
            effect_clean = clean_feature_name(insight.get("effect", "Y"))
            lag = insight.get("lag_days", 1)
            questions = [
                f"What physical or process bottlenecks cause the {lag}-day lag between {cause_clean} and {effect_clean}?",
                f"Does this delay shorten or lengthen during high-volume periods or holidays?",
                f"How can we operationalize {cause_clean} as an early warning system to prepare for {effect_clean} shifts?"
            ]
            
        return questions

    def generate_anomalies(self, df: pd.DataFrame) -> list:
        insights = []
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            mean = numeric_df[col].mean()
            std = numeric_df[col].std()
            if pd.isna(std) or std == 0:
                continue
                
            threshold = mean + 3 * std
            anomalies = df[numeric_df[col] > threshold]
            if not anomalies.empty:
                date_str = "recent periods"
                if "date" in df.columns:
                    date_val = anomalies["date"].iloc[0]
                    date_str = str(date_val.date() if hasattr(date_val, 'date') else date_val)
                elif df.index.name == "date" or pd.api.types.is_datetime64_any_dtype(df.index):
                    date_val = anomalies.index[0]
                    date_str = str(date_val.date() if hasattr(date_val, 'date') else date_val)
                
                desc = f"Anomaly detected in {col} exceeding 3-sigma."
                insights.append({
                    "type": "anomaly",
                    "col": col,
                    "description": desc,
                    "hypothesis": f"Hypothesis: External shock caused {col} to spike.",
                    "questions": [
                        f"Why does {col} spike around {date_str}?",
                        f"What external event could explain this extreme variation?"
                    ]
                })
        return insights


    def generate_all(self, insights: list, df: pd.DataFrame = None) -> list:
        logger.info("Generating questions...")
        for insight in insights:
            insight["questions"] = self.generate(insight)
            
        if df is not None:
            anomaly_insights = self.generate_anomalies(df)
            insights.extend(anomaly_insights)
            
        return insights

def run_pipeline(filepath: str):
    pass
