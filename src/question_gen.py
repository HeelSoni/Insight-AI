import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class QuestionGenerator:
    def __init__(self):
        pass

    def generate(self, insight: dict) -> list:
        itype = insight.get("type")
        questions = []
        
        if itype == "correlation":
            col_a = insight.get("col_a", "A")
            col_b = insight.get("col_b", "B")
            questions = [
                f"Why does {col_a} change significantly when {col_b} fluctuates?",
                f"Is this relationship consistent across all time periods, or only seasonally?",
                f"Which other variables might be driving both {col_a} and {col_b}?"
            ]

        elif itype == "cluster":
            c_int = insight.get("cluster_id", 0)
            questions = [
                f"Why is Cluster {c_int} behaving so differently from the others?",
                f"What event or factor could explain the unique pattern in Cluster {c_int}?",
                f"How has the size of Cluster {c_int} changed over time?"
            ]

        elif itype == "association":
            ant = insight.get("antecedent", "A")
            con = insight.get("consequent", "B")
            questions = [
                f"Why does {ant} so strongly predict {con}?",
                f"Does this association hold in all segments, or only certain groups?",
                f"What would happen to {con} if {ant} were discontinued or reduced?"
            ]

        elif itype == "time_lag":
            cause = insight.get("cause", "X")
            effect = insight.get("effect", "Y")
            lag = insight.get("lag_days", 1)
            questions = [
                f"Why is there a {lag}-day delay between {cause} and {effect}?",
                f"Does this delay shorten during peak periods or remain constant?",
                f"Can {cause} be used as a reliable early-warning signal for {effect}?"
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
