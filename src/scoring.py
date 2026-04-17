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
            return min((lift - 1) * 25, 100)
        elif itype == "time_lag":
            r = abs(insight.get("r_value", 0))
            return min(r * 100, 100)
        elif itype == "anomaly":
            return 90.0
        return 50.0

    def _score_novelty(self, insight: dict) -> float:
        score = 60.0
        
        cols_involved = []
        if "col_a" in insight: cols_involved.extend([insight.get("col_a", ""), insight.get("col_b", "")])
        elif "antecedent" in insight: cols_involved.extend([insight.get("antecedent", ""), insight.get("consequent", "")])
        elif "cause" in insight: cols_involved.extend([insight.get("cause", ""), insight.get("effect", "")])
        elif "col" in insight: cols_involved.append(insight.get("col", ""))
        elif "defining_features" in insight: cols_involved.extend(insight.get("defining_features", {}).keys())
        
        cols_involved = [str(c).lower() for c in cols_involved]
        
        for i in range(len(cols_involved)):
            for j in range(i+1, len(cols_involved)):
                words_i = set(cols_involved[i].split('_'))
                words_j = set(cols_involved[j].split('_'))
                if words_i.intersection(words_j):
                    score -= 30
                    break
                    
        money_words = {'usd', 'price', 'sales', 'revenue', 'spend', 'profit'}
        weather_words = {'temp', 'temperature', 'rain', 'rainfall', 'weather', 'sun'}
        
        has_money = any(any(w in c for w in money_words) for c in cols_involved)
        has_weather = any(any(w in c for w in weather_words) for c in cols_involved)
        
        if has_money and has_weather:
            score += 20
            
        return max(0.0, min(score, 100.0))

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
            return conf * 100
        elif itype == "time_lag":
            r = abs(insight.get("r_value", 0))
            return r * 80
        elif itype == "anomaly":
            return 85.0
        return 50.0

    def score_all(self, insights: list) -> list:
        logger.info("Scoring insights...")
        for insight in insights:
            conf = self._score_confidence(insight)
            nov = self._score_novelty(insight)
            imp = self._score_impact(insight)
            
            final_score = (0.4 * conf) + (0.3 * nov) + (0.3 * imp)
            
            insight["confidence"] = float(conf)
            insight["novelty"] = float(nov)
            insight["impact"] = float(imp)
            insight["final_score"] = float(final_score)
            
        return sorted(insights, key=lambda x: x.get("final_score", 0), reverse=True)

def run_pipeline(filepath: str):
    pass
