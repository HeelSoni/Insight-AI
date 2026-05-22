import logging

logger = logging.getLogger(__name__)


class InsightScorer:
    def __init__(self):
        pass

    def _score_confidence(self, insight: dict) -> float:
        itype = insight.get("type")

        if itype == "correlation":
            r = abs(insight.get("r_value", 0))
            return min(r * 100, 100)

        elif itype == "cluster":
            sil = insight.get("silhouette_score", 0)
            return max(0, min(sil * 100, 100))

        elif itype == "association":
            lift = insight.get("lift", 1)
            conf = insight.get("confidence", 0)
            # Blend lift and confidence for a balanced view
            return min(((lift - 1) * 20) + (conf * 50), 100)

        elif itype == "time_lag":
            r = abs(insight.get("r_value", 0))
            return min(r * 100, 100)

        elif itype == "anomaly":
            # Anomalies are statistically certain by definition (3-sigma)
            return 90.0

        elif itype == "distribution":
            # Higher dominance = higher confidence the pattern is real
            top_pct = insight.get("top_pct", 50)
            return min(top_pct, 100)

        elif itype == "numeric_insight":
            skew = abs(insight.get("skew", 0))
            cv   = insight.get("cv", 0)
            if skew:
                return min(skew * 30 + 40, 100)
            if cv:
                return min(cv * 0.5 + 40, 100)
            return 60.0

        return 50.0

    def _score_novelty(self, insight: dict) -> float:
        itype = insight.get("type")

        # Base novelty by type — time lags and anomalies are inherently novel
        base = {
            "time_lag":        80.0,
            "anomaly":         75.0,
            "cluster":         70.0,
            "association":     65.0,
            "numeric_insight": 60.0,
            "correlation":     55.0,
            "distribution":    50.0,
        }.get(itype, 55.0)

        # Boost if insight involves many columns / unique dimensions
        if itype == "cluster":
            n_features = len(insight.get("defining_features", {}))
            base += min(n_features * 5, 15)

        if itype == "association":
            # Multi-factor rules are more novel
            ant = insight.get("antecedent", "")
            con = insight.get("consequent", "")
            n_factors = ant.count(",") + con.count(",") + 2
            base += min(n_factors * 4, 20)

        return max(0.0, min(base, 100.0))

    def _score_impact(self, insight: dict) -> float:
        itype = insight.get("type")

        if itype == "correlation":
            r = abs(insight.get("r_value", 0))
            return r * 100

        elif itype == "cluster":
            size = insight.get("size_pct", 0)
            return min(size * 2, 100)

        elif itype == "association":
            conf = insight.get("confidence", 0)
            lift = insight.get("lift", 1)
            return min(conf * 60 + (lift - 1) * 10, 100)

        elif itype == "time_lag":
            r = abs(insight.get("r_value", 0))
            return r * 80

        elif itype == "anomaly":
            count = insight.get("outlier_count", 1)
            return min(85 + count * 2, 100)

        elif itype == "distribution":
            top_pct  = insight.get("top_pct", 50)
            n_unique = insight.get("n_unique", 2)
            # More skewed + more categories = higher impact
            return min(top_pct * 0.5 + n_unique * 3, 100)

        elif itype == "numeric_insight":
            skew = abs(insight.get("skew", 0))
            cv   = insight.get("cv", 0)
            if skew:
                return min(skew * 25 + 30, 100)
            if cv:
                return min(cv * 0.4 + 30, 100)
            return 50.0

        return 50.0

    def score_all(self, insights: list) -> list:
        logger.info("Scoring insights...")
        for insight in insights:
            conf = self._score_confidence(insight)
            nov  = self._score_novelty(insight)
            imp  = self._score_impact(insight)

            final_score = (0.4 * conf) + (0.3 * nov) + (0.3 * imp)

            insight["confidence"]  = float(conf)
            insight["novelty"]     = float(nov)
            insight["impact"]      = float(imp)
            insight["final_score"] = float(final_score)

        return sorted(insights, key=lambda x: x.get("final_score", 0), reverse=True)


def run_pipeline(filepath: str):
    pass
