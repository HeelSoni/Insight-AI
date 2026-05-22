import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
from src import preprocessing, pattern_engine, hypothesis, question_gen, simulator, scoring, formatting

importlib.reload(preprocessing)
importlib.reload(pattern_engine)
importlib.reload(hypothesis)
importlib.reload(question_gen)
importlib.reload(simulator)
importlib.reload(scoring)
importlib.reload(formatting)

from src.preprocessing import DataPreprocessor
from src.pattern_engine import PatternEngine
from src.hypothesis import HypothesisGenerator
from src.question_gen import QuestionGenerator
from src.simulator import CounterfactualSimulator
from src.scoring import InsightScorer
from src.formatting import clean_feature_name, clean_ohe_value, md_to_html


st.set_page_config(page_title="Insight AI", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Better UI ---
st.markdown("""
<style>
    /* Global Styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    /* Insight Cards */
    .insight-card {
        background-color: #1E1E24;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border-left: 5px solid #FF416C;
    }
    .insight-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #fff;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-right: 10px;
        margin-bottom: 15px;
    }
    .badge-correlation { background-color: #4CAF50; color: white; }
    .badge-cluster { background-color: #2196F3; color: white; }
    .badge-association { background-color: #9C27B0; color: white; }
    .badge-timelag { background-color: #FF9800; color: white; }
    .badge-anomaly { background-color: #F44336; color: white; }
    .badge-distribution { background-color: #00BCD4; color: white; }
    .badge-numericinsight { background-color: #607D8B; color: white; }
    
    .score-badge { background-color: #333; color: #FFA500; }
    
    .section-title {
        font-size: 1.1rem;
        color: #bbb;
        margin-top: 15px;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .text-content {
        font-size: 1rem;
        color: #e0e0e0;
        line-height: 1.5;
    }
    
    /* Metrics Override */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #FF416C;
    }
    
    /* Scenario Cards */
    .scenario-card {
        background-color: #1E1E24;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border-left: 5px solid #00C9FF;
    }
    .scenario-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #fff;
        margin-bottom: 10px;
    }
    
    /* Micro callout cards */
    .micro-card {
        background-color: #1E1E24;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #FF416C;
        color: #e0e0e0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-title">Insight AI ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous Data Discovery & Reasoning System</div>', unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103322.png", width=60)
    st.markdown("### Control Panel")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
    st.divider()
    num_insights_to_show = st.slider("Insights to Show", 1, 50, 10, help="Number of top insights generated")
    min_score = st.slider("Min Confidence Score", 0, 100, 50)
    st.markdown("___")
    st.caption("Insight AI engines detect correlations, clusters, logic-rules, and time-shifted phenomena entirely automatically.")


@st.cache_data(show_spinner=False)
def process_data_v4(file_obj_or_path):
    preprocessor = DataPreprocessor()
    
    # Bug fix: Streamlit UploadedFile requires seeking to 0 if it's read by pd.read_csv
    if hasattr(file_obj_or_path, 'seek'):
        file_obj_or_path.seek(0)
        
    df = preprocessor.load_data(file_obj_or_path)
    
    # We call individual functions to prevent load_data from being fired again with an exhausted buffer
    df_clean = preprocessor.clean(df)
    df_encoded = preprocessor.encode(df_clean)
    prof = preprocessor.profile(df_encoded)
    
    for col in df_clean.select_dtypes(include=['datetime64[ns]']).columns:
        if col in df_encoded.columns:
            df_encoded[col] = df_clean[col]
            
    pattern_eng = PatternEngine()
    hypo_gen = HypothesisGenerator()
    q_gen = QuestionGenerator()
    scorer = InsightScorer()
    
    insights = pattern_eng.discover(df_encoded)
    insights = hypo_gen.generate_all(insights)
    insights = q_gen.generate_all(insights, df_encoded)
    insights = scorer.score_all(insights)
    
    return df, df_encoded, prof, insights


# If no file is uploaded, try loading sample
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_sales_weather.csv')

with st.spinner("Analyzing data patterns... please wait."):
    if uploaded_file is not None:
        df_raw, df_proc, profile, all_insights = process_data_v4(uploaded_file)
    elif os.path.exists(data_path):
        df_raw, df_proc, profile, all_insights = process_data_v4(data_path)
    else:
        st.info("Please upload a CSV or generate the sample data via terminal (`python data/generate_data.py`).")
        st.stop()

# ─── Enforce type diversity ───────────────────────────────────────────────────
# Never let any single insight type dominate — cap each type at 3 slots.
# This guarantees the Top Insights list is always a rich, varied cross-section.
from collections import defaultdict

def _build_diverse_insights(pool: list, max_per_type: int, total: int) -> list:
    """Pick insights from pool ensuring no type appears more than max_per_type times."""
    type_counts: dict = defaultdict(int)
    diverse: list = []
    # First pass: fill up to max_per_type per type
    for ins in pool:
        t = ins.get("type", "unknown")
        if type_counts[t] < max_per_type:
            diverse.append(ins)
            type_counts[t] += 1
        if len(diverse) >= total:
            break
    # Second pass: if still short, fill with anything left
    if len(diverse) < total:
        seen_ids = set(id(i) for i in diverse)
        for ins in pool:
            if id(ins) not in seen_ids:
                diverse.append(ins)
                seen_ids.add(id(ins))
            if len(diverse) >= total:
                break
    return diverse

# Filter by minimum score first, then apply diversity cap
_filtered_pool = [i for i in all_insights if i.get("final_score", 0) >= min_score]
insights = _build_diverse_insights(_filtered_pool, max_per_type=3, total=num_insights_to_show)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💡 Top Insights", "🎛️ Simulate", "📈 Data Explorer", "⚙️ Raw Patterns"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{profile['shape'][0]:,}")
    with col2:
        st.metric("Total Features", profile['shape'][1])
    with col3:
        st.metric("Numeric Cols", len(profile["numeric_cols"]))
    with col4:
        st.metric("Insights Discovered", len(all_insights))
        
    st.divider()
    
    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        st.subheader("Correlation Heatmap")
        corr = df_proc[profile["numeric_cols"]].corr()
        fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig_corr.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Explain the Correlation Heatmap
        st.markdown(f"""
        <div style="background-color: #1E1E24; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #FF416C;">
            <p style="margin: 0; font-weight: bold; color: #fff; font-size: 1.05rem;">
                💡 Understanding the Correlation Heatmap
            </p>
            <p style="margin: 6px 0 0 0; color: #ccc; font-size: 0.9rem; line-height: 1.45;">
                This heatmap represents the strength of linear relationships between your numeric variables. 
                Scores range from <strong>-1.00</strong> (perfect inverse relationship) to <strong>+1.00</strong> (perfect direct relationship). 
                A score of <strong>0.00</strong> implies the variables move entirely independently of one another.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate dynamic insights
        strong_pos = []
        strong_neg = []
        for col_x in corr.columns:
            for col_y in corr.columns:
                if col_x != col_y:
                    val = corr.loc[col_x, col_y]
                    if val > 0.45:
                        strong_pos.append((col_x, col_y, val))
                    elif val < -0.45:
                        strong_neg.append((col_x, col_y, val))
                        
        unique_pos = []
        for x, y, v in sorted(strong_pos, key=lambda item: item[2], reverse=True):
            if not any((y == ux and x == uy) for ux, uy, _ in unique_pos):
                unique_pos.append((x, y, v))
                
        unique_neg = []
        for x, y, v in sorted(strong_neg, key=lambda item: item[2]):
            if not any((y == ux and x == uy) for ux, uy, _ in unique_neg):
                unique_neg.append((x, y, v))
                
        st.markdown("<p style='margin-top:15px; margin-bottom:5px; font-weight:bold; color:#fff;'>🔍 What this tells us about your active dataset:</p>", unsafe_allow_html=True)
        if unique_pos or unique_neg:
            for x, y, v in unique_pos[:2]:
                cx = clean_feature_name(x)
                cy = clean_feature_name(y)
                st.markdown(
                    f'<div class="micro-card" style="border-left: 4px solid #4CAF50;">'
                    f'📈 <strong>Direct Co-Movement</strong>: <strong>{cx}</strong> and <strong>{cy}</strong> move in the same direction with a positive correlation of <strong>{v:.2f}</strong>. Elevating one will likely drive the other upward.'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            for x, y, v in unique_neg[:2]:
                cx = clean_feature_name(x)
                cy = clean_feature_name(y)
                st.markdown(
                    f'<div class="micro-card" style="border-left: 4px solid #F44336;">'
                    f'📉 <strong>Inverse Constraint</strong>: <strong>{cx}</strong> and <strong>{cy}</strong> move in opposite directions with a negative correlation of <strong>{v:.2f}</strong>. Increases in one correspond to drops in the other.'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f'<div class="micro-card" style="border-left: 4px solid #FFA500;">'
                f'ℹ️ No strong linear correlations (above &plusmn;0.45) detected in this dataset. The numeric fields seem to operate independently.'
                f'</div>',
                unsafe_allow_html=True
            )
        
    with col_b:
        st.subheader("Distribution Scatter Matrix")
        st.caption("Top 4 Numeric Features")
        top_cols = profile["numeric_cols"][:4]
        if len(top_cols) > 1:
            fig_pair = px.scatter_matrix(df_raw, dimensions=top_cols, template="plotly_dark")
            fig_pair.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(color='#FF416C', opacity=0.6))
            fig_pair.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=500)
            st.plotly_chart(fig_pair, use_container_width=True)
            
            # Explain the Scatter Matrix
            st.markdown(f"""
            <div style="background-color: #1E1E24; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #00C9FF;">
                <p style="margin: 0; font-weight: bold; color: #fff; font-size: 1.05rem;">
                    💡 Understanding the Scatter Matrix
                </p>
                <p style="margin: 6px 0 0 0; color: #ccc; font-size: 0.9rem; line-height: 1.45;">
                    This matrix compares pairs of variables against each other in scatter plots. 
                    Look for <strong>clustering patterns</strong> (groups of points close together), <strong>slopes or curves</strong> (trends), 
                    or <strong>diffuse clouds</strong> (indicating random distribution).
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<p style='margin-top:15px; margin-bottom:5px; font-weight:bold; color:#fff;'>🔍 What this tells us about your active dataset:</p>", unsafe_allow_html=True)
            for col in top_cols[:2]:
                col_clean = clean_feature_name(col)
                skew = df_proc[col].skew()
                if abs(skew) > 0.8:
                    direction = "skewed toward high values (right tail)" if skew > 0 else "skewed toward low values (left tail)"
                    st.markdown(f"📊 **{col_clean}** exhibits high skewness, concentrating heavily on the **{direction}**. Watch out for outliers affecting baseline averages.")
                else:
                    st.markdown(f"📊 **{col_clean}** shows a relatively balanced, even distribution across its sample scale.")
        else:
            st.info("Not enough numeric columns for pairplot.")

def get_badge_class(itype):
    t = itype.lower().replace(' ', '')
    if 'correlation' in t: return 'badge-correlation'
    if 'cluster' in t: return 'badge-cluster'
    if 'association' in t: return 'badge-association'
    if 'lag' in t: return 'badge-timelag'
    if 'anomaly' in t: return 'badge-anomaly'
    if 'distribution' in t: return 'badge-distribution'
    if 'numeric' in t: return 'badge-numericinsight'
    return 'badge-distribution'

with tab2:
    st.subheader(f"Top {len(insights)} Deep Insights")
    if not insights:
         st.warning("No insights found satisfying the given minimum score threshold.")
         
    for idx, insight in enumerate(insights):
        score = insight.get("final_score", 0)
        itype = insight.get("type", "Unknown").replace("_", " ").title()
        desc = insight.get("description", "")
        hypo = insight.get("hypothesis", "")
        qs = insight.get("questions", [])
        
        c_score = insight.get('confidence', 0)
        n_score = insight.get('novelty', 0)
        i_score = insight.get('impact', 0)
        
        badge_cls = get_badge_class(itype)
        
        # Convert markdown parameters in descriptors to html tags
        desc_html = md_to_html(desc)
        hypo_html = md_to_html(hypo)
        questions_html = "".join([f"<li>{md_to_html(q)}</li>" for q in qs])
        
        html_str = (
            f'<div class="insight-card">'
            f'<div class="insight-title">🔍 Insight #{idx+1}</div>'
            f'<span class="badge {badge_cls}">{itype}</span>'
            f'<span class="badge score-badge">Score: {score:.1f} (Conf: {c_score:.0f} | Nov: {n_score:.0f} | Imp: {i_score:.0f})</span>'
            f'<div class="section-title">💡 The Pattern</div>'
            f'<div class="text-content">{desc_html}</div>'
            f'<div class="section-title">🧠 Business Hypothesis & Reasoning</div>'
            f'<div class="text-content">{hypo_html}</div>'
            f'<div class="section-title">🙋 Strategic Next-Step Questions</div>'
            f'<div class="text-content"><ul>{questions_html}</ul></div>'
            f'</div>'
        )
        st.markdown(html_str, unsafe_allow_html=True)


with tab3:
    st.header("🎛️ Counterfactual Simulator")
    st.markdown("Use Machine Learning to simulate *What-If* scenarios by altering historical signals.")

    sim = CounterfactualSimulator()
    top_sims = sim.simulate_top_pairs(df_proc)

    # ── Helper to pick accent color based on delta ──────────────────────────────
    def _sim_accent(delta_pct):
        if delta_pct > 2:   return "#4CAF50"   # green  – positive
        if delta_pct < -2:  return "#F44336"   # red    – risky
        return "#FFA500"                        # orange – neutral

    # ── Auto-Simulated Scenarios ────────────────────────────────────────────────
    if top_sims:
        with st.expander("✨ Auto-Simulated Recommended Scenarios", expanded=True):
            cols_sim = st.columns(len(top_sims))
            for s_idx, ts in enumerate(top_sims):
                with cols_sim[s_idx]:
                    accent    = _sim_accent(ts.get("delta_pct", 0))
                    narrative = ts.get("narrative", "")
                    reco      = ts.get("recommendation", "")
                    delta     = ts.get("delta_pct", 0)
                    orig      = ts.get("predicted_original_mean", 0)
                    mod       = ts.get("predicted_modified_mean", 0)
                    clean_t   = clean_feature_name(ts.get("target_col", ""))
                    clean_c   = clean_feature_name(ts.get("change_col", ""))
                    direction_icon = "📈" if delta > 0 else "📉"

                    st.markdown(
                        f'<div style="background:#1E1E24; border-radius:12px; padding:18px; '
                        f'border-left:5px solid {accent}; margin-bottom:8px;">'
                        f'<p style="font-size:1.1rem; font-weight:700; color:#fff; margin:0 0 8px 0;">'
                        f'Scenario #{s_idx+1}</p>'
                        f'<p style="font-size:0.88rem; color:#bbb; margin:0 0 10px 0;">'
                        f'{md_to_html(narrative)}</p>'
                        f'<div style="display:flex; gap:12px; margin-bottom:10px;">'
                        f'<div style="background:#111; border-radius:8px; padding:8px 14px; flex:1; text-align:center;">'
                        f'<div style="font-size:0.75rem; color:#888;">Baseline</div>'
                        f'<div style="font-size:1.1rem; font-weight:700; color:#fff;">{orig:.2f}</div>'
                        f'</div>'
                        f'<div style="background:#111; border-radius:8px; padding:8px 14px; flex:1; text-align:center;">'
                        f'<div style="font-size:0.75rem; color:#888;">Simulated</div>'
                        f'<div style="font-size:1.1rem; font-weight:700; color:{accent};">{mod:.2f}</div>'
                        f'</div>'
                        f'<div style="background:#111; border-radius:8px; padding:8px 14px; flex:1; text-align:center;">'
                        f'<div style="font-size:0.75rem; color:#888;">Shift</div>'
                        f'<div style="font-size:1.1rem; font-weight:700; color:{accent};">{direction_icon} {delta:+.1f}%</div>'
                        f'</div>'
                        f'</div>'
                        f'<p style="font-size:0.83rem; color:#aaa; margin:0;">{md_to_html(reco)}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    st.divider()

    # ── Custom Simulation Run ────────────────────────────────────────────────────
    st.subheader("Custom Simulation Run")
    sim_num_cols = profile["numeric_cols"]
    if len(sim_num_cols) < 2:
        st.warning("Need at least 2 numeric columns to run simulations.")
    else:
        col_target, col_change, col_val = st.columns(3)
        target_col = col_target.selectbox("Target Outcome Predictor", sim_num_cols, index=0, format_func=clean_feature_name)
        change_col = col_change.selectbox("Influential Variable to Modify", sim_num_cols, index=1, format_func=clean_feature_name)
        pct_change = col_val.slider("% Intervention Change", -50, 100, 10)

        if st.button("🚀 Run AI Simulation"):
            with st.spinner("Running Random Forest Simulation..."):
                time.sleep(0.5)
                res = sim.simulate(df_proc, target_col, change_col, pct_change)

            if "error" in res:
                st.error(res["error"])
            else:
                narrative      = res.get("narrative", "")
                recommendation = res.get("recommendation", "")
                delta_pct      = res.get("delta_pct", 0)
                pred_orig      = res.get("predicted_original_mean", 0)
                pred_mod       = res.get("predicted_modified_mean", 0)
                accent         = _sim_accent(delta_pct)
                direction_icon = "📈" if delta_pct > 0 else "📉"
                clean_t        = clean_feature_name(target_col)
                clean_c        = clean_feature_name(change_col)

                # Narrative card
                if narrative:
                    st.markdown(
                        f'<div style="background:#1E1E24; border-radius:10px; padding:16px 20px; '
                        f'border-left:5px solid {accent}; margin-bottom:12px;">'
                        f'<p style="margin:0; font-size:1rem; color:#e0e0e0;">{md_to_html(narrative)}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Recommendation card
                if recommendation:
                    st.markdown(
                        f'<div style="background:#12293a; border-radius:10px; padding:14px 18px; '
                        f'border-left:5px solid #00C9FF; margin-bottom:18px;">'
                        f'<p style="margin:0; font-size:0.95rem; color:#cce8ff;">{md_to_html(recommendation)}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.divider()
                metric_col, chart_col = st.columns([1, 1], gap="large")

                with metric_col:
                    # KPI tiles
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Baseline Avg", f"{pred_orig:.2f}")
                    k2.metric("Simulated Avg", f"{pred_mod:.2f}", f"{pred_mod - pred_orig:+.2f}")
                    k3.metric("Predicted Shift", f"{delta_pct:+.2f}%")

                    chart_data = pd.DataFrame({
                        "Scenario": ["Actual Baseline", "Simulated Intervention"],
                        "Predicted Mean": [pred_orig, pred_mod]
                    })
                    fig_bar = px.bar(
                        chart_data, x="Scenario", y="Predicted Mean", color="Scenario",
                        color_discrete_sequence=["#555566", accent],
                        text_auto='.2f', template="plotly_dark",
                        title=f"Impact on {clean_t}"
                    )
                    fig_bar.update_layout(
                        showlegend=False, height=320,
                        margin=dict(l=0, r=0, t=40, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e0e0')
                    )
                    fig_bar.update_traces(marker_line_width=0, opacity=0.9)
                    st.plotly_chart(fig_bar, use_container_width=True)

                with chart_col:
                    st.markdown(
                        '<p style="font-weight:700; font-size:1rem; margin-bottom:4px;">🔑 Key Predictive Drivers</p>'
                        '<p style="font-size:0.85rem; color:#888; margin-bottom:12px;">'
                        'Relative importance of each feature as detected by the AI model.</p>',
                        unsafe_allow_html=True
                    )
                    for driver in res.get("drivers", []):
                        imp   = driver.get("importance", 0)
                        feat  = driver.get("feature", "")
                        bar_w = max(1, int(imp))
                        bar_color = accent if imp >= 20 else "#00C9FF"
                        st.markdown(
                            f'<div style="margin-bottom:10px;">'
                            f'<div style="display:flex; justify-content:space-between; margin-bottom:3px;">'
                            f'<span style="font-size:0.9rem; color:#e0e0e0;">{feat}</span>'
                            f'<span style="font-size:0.85rem; color:{bar_color}; font-weight:600;">{imp:.1f}%</span>'
                            f'</div>'
                            f'<div style="background:#2a2a2a; border-radius:4px; height:8px;">'
                            f'<div style="background:{bar_color}; width:{bar_w}%; height:8px; border-radius:4px;"></div>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )


with tab4:
    st.header("📈 Interactive Data Explorer")
    st.markdown("Build beautiful charts by choosing any columns — the AI will aggregate data intelligently for you.")

    # ── Chart type ──────────────────────────────────────────────────────────────
    CHART_TYPES = ["Scatter Plot", "Line Chart", "Bar Chart (Aggregated)", "Box Plot", "Histogram", "Pie Chart (Aggregated)", "Correlation Heatmap"]
    chart_type = st.selectbox("📊 Select Chart Type", CHART_TYPES, key="explorer_chart_type")

    # ── Per-chart explanation ────────────────────────────────────────────────────
    CHART_TIPS = {
        "Scatter Plot":               "Shows the relationship between two numeric columns. Look for trends, clusters, or outliers.",
        "Line Chart":                 "Tracks how a numeric value changes along an ordered axis (e.g. time, episode, season).",
        "Bar Chart (Aggregated)":     "Groups rows by a category and computes the **mean** of a numeric column per group.",
        "Box Plot":                   "Displays the spread and median of a numeric column per category. Outliers appear as dots.",
        "Histogram":                  "Shows the distribution of a single numeric column — how often each value range occurs.",
        "Pie Chart (Aggregated)":     "Shows the **sum** of a numeric column broken down by a categorical column.",
        "Correlation Heatmap":        "Displays the strength of linear relationships between all numeric columns at once.",
    }
    st.markdown(
        f'<div class="micro-card" style="border-left:4px solid #00C9FF; margin-bottom:16px;">'
        f'💡 <strong>{chart_type}</strong>: {CHART_TIPS[chart_type]}'
        f'</div>',
        unsafe_allow_html=True
    )

    num_cols  = profile["numeric_cols"]
    cat_cols  = profile["categorical_cols"]
    all_cols  = num_cols + cat_cols
    custom_colors = ['#FF416C', '#00C9FF', '#92FE9D', '#FFB75E', '#8A2387', '#E94057',
                     '#43E97B', '#FA709A', '#FEE140', '#30CFD0']

    def _base_layout(height=580):
        return dict(
            height=height,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, l=20, r=20, b=20),
            font=dict(color='#e0e0e0', family="Inter, sans-serif"),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#2a2a2a', zeroline=False),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#2a2a2a', zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

    # ── Correlation Heatmap (no column pickers needed) ───────────────────────────
    if chart_type == "Correlation Heatmap":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")
        else:
            corr_df = df_raw[num_cols].corr()
            clean_labels = [clean_feature_name(c) for c in corr_df.columns]
            fig = px.imshow(
                corr_df, x=clean_labels, y=clean_labels,
                text_auto='.2f', aspect="auto",
                color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                template="plotly_dark"
            )
            fig.update_layout(**_base_layout(600))
            st.plotly_chart(fig, use_container_width=True)

    # ── Histogram (single column) ────────────────────────────────────────────────
    elif chart_type == "Histogram":
        c1, c2 = st.columns(2)
        h_col   = c1.selectbox("Column to Distribute", num_cols, key="hist_col")
        h_color = c2.selectbox("Color By (Optional)", ["None"] + cat_cols, key="hist_color")
        color_val = None if h_color == "None" else h_color

        fig = px.histogram(
            df_raw, x=h_col, color=color_val,
            template="plotly_dark",
            color_discrete_sequence=custom_colors,
            nbins=30,
            title=f"Distribution of {clean_feature_name(h_col)}"
        )
        fig.update_traces(opacity=0.85, marker_line_width=1, marker_line_color='#111')
        fig.update_layout(**_base_layout())
        st.plotly_chart(fig, use_container_width=True)

    # ── Scatter Plot ─────────────────────────────────────────────────────────────
    elif chart_type == "Scatter Plot":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for a scatter plot.")
        else:
            c1, c2, c3 = st.columns(3)
            sx = c1.selectbox("X-Axis", num_cols, index=0, key="sc_x")
            sy = c2.selectbox("Y-Axis", num_cols, index=1, key="sc_y")
            sc = c3.selectbox("Color By (Optional)", ["None"] + cat_cols, key="sc_c")
            color_val = None if sc == "None" else sc

            fig = px.scatter(
                df_raw, x=sx, y=sy, color=color_val,
                template="plotly_dark",
                color_discrete_sequence=custom_colors,
                title=f"{clean_feature_name(sx)} vs {clean_feature_name(sy)}",
                hover_data=df_raw.columns.tolist()
            )
            fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color='#111')))
            fig.update_layout(**_base_layout())
            st.plotly_chart(fig, use_container_width=True)

    # ── Line Chart ───────────────────────────────────────────────────────────────
    elif chart_type == "Line Chart":
        if len(num_cols) < 1:
            st.warning("Need at least 1 numeric column for a line chart.")
        else:
            c1, c2, c3 = st.columns(3)
            lx = c1.selectbox("X-Axis (ordered)", all_cols, index=0, key="ln_x")
            ly = c2.selectbox("Y-Axis (numeric)", num_cols, index=min(1, len(num_cols)-1), key="ln_y")
            lc = c3.selectbox("Color By (Optional)", ["None"] + cat_cols, key="ln_c")
            color_val = None if lc == "None" else lc

            try:
                df_line = df_raw.sort_values(by=lx)
            except Exception:
                df_line = df_raw

            fig = px.line(
                df_line, x=lx, y=ly, color=color_val,
                template="plotly_dark",
                color_discrete_sequence=custom_colors,
                title=f"{clean_feature_name(ly)} over {clean_feature_name(lx)}"
            )
            fig.update_traces(line=dict(width=2.5))
            fig.update_layout(**_base_layout())
            st.plotly_chart(fig, use_container_width=True)

    # ── Bar Chart (Aggregated) ───────────────────────────────────────────────────
    elif chart_type == "Bar Chart (Aggregated)":
        if not cat_cols:
            st.warning("No categorical columns detected. Bar charts work best with a category on the X-axis.")
        elif not num_cols:
            st.warning("No numeric columns detected for the Y-axis.")
        else:
            c1, c2, c3 = st.columns(3)
            bx = c1.selectbox("X-Axis (Category)", cat_cols, index=0, key="bar_x")
            by = c2.selectbox("Y-Axis (Mean of)", num_cols, index=0, key="bar_y")
            bc = c3.selectbox("Color By (Optional)", ["None"] + cat_cols, key="bar_c")
            color_val = None if bc == "None" else bc

            agg_df = df_raw.groupby(bx if not color_val else [bx, color_val], as_index=False)[by].mean()
            agg_df.rename(columns={by: f"Mean {clean_feature_name(by)}"}, inplace=True)
            y_label = f"Mean {clean_feature_name(by)}"

            fig = px.bar(
                agg_df, x=bx, y=y_label, color=color_val or bx,
                template="plotly_dark",
                color_discrete_sequence=custom_colors,
                text_auto='.2f',
                title=f"Average {clean_feature_name(by)} by {clean_feature_name(bx)}"
            )
            fig.update_traces(marker_line_width=0, opacity=0.92)
            fig.update_layout(**_base_layout())
            st.plotly_chart(fig, use_container_width=True)

    # ── Box Plot ─────────────────────────────────────────────────────────────────
    elif chart_type == "Box Plot":
        c1, c2, c3 = st.columns(3)
        bx_x = c1.selectbox("Group By (Category)", ["None"] + cat_cols, key="box_x")
        bx_y = c2.selectbox("Value Column (Numeric)", num_cols, index=0, key="box_y")
        bx_c = c3.selectbox("Color By (Optional)", ["None"] + cat_cols, key="box_c")
        x_val     = None if bx_x == "None" else bx_x
        color_val = None if bx_c == "None" else bx_c

        fig = px.box(
            df_raw, x=x_val, y=bx_y, color=color_val,
            template="plotly_dark",
            color_discrete_sequence=custom_colors,
            points="outliers",
            title=f"Distribution of {clean_feature_name(bx_y)}"
                  + (f" by {clean_feature_name(bx_x)}" if x_val else "")
        )
        fig.update_layout(**_base_layout())
        st.plotly_chart(fig, use_container_width=True)

    # ── Pie Chart (Aggregated) ───────────────────────────────────────────────────
    elif chart_type == "Pie Chart (Aggregated)":
        if not cat_cols:
            st.warning("No categorical columns detected. Pie charts need a category column for slices.")
        elif not num_cols:
            st.warning("No numeric columns detected for slice values.")
        else:
            c1, c2 = st.columns(2)
            pc_names  = c1.selectbox("Slice Categories", cat_cols, index=0, key="pie_names")
            pc_values = c2.selectbox("Slice Values (Sum of)", num_cols, index=0, key="pie_vals")

            pie_df = df_raw.groupby(pc_names, as_index=False)[pc_values].sum()
            # Keep top-10 slices, group rest as "Other"
            if len(pie_df) > 10:
                top10 = pie_df.nlargest(10, pc_values)
                other_sum = pie_df[~pie_df[pc_names].isin(top10[pc_names])][pc_values].sum()
                other_row = pd.DataFrame({pc_names: ["Other"], pc_values: [other_sum]})
                pie_df = pd.concat([top10, other_row], ignore_index=True)

            fig = px.pie(
                pie_df, names=pc_names, values=pc_values,
                template="plotly_dark",
                color_discrete_sequence=custom_colors,
                hole=0.42,
                title=f"Share of {clean_feature_name(pc_values)} by {clean_feature_name(pc_names)}"
            )
            fig.update_traces(
                textposition='outside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#1E1E24', width=2)),
                pull=[0.03] * len(pie_df)
            )
            lay = _base_layout(520)
            lay.pop("xaxis", None); lay.pop("yaxis", None)
            fig.update_layout(**lay)
            st.plotly_chart(fig, use_container_width=True)


with tab5:
    st.header("⚙️ Raw Pipeline Details")
    st.write("Browse through all discovered vectors beneath the hood.")
    df_insights = pd.DataFrame(all_insights)
    if not df_insights.empty:
        # Strip Markdown bold asterisks out for raw tabular display
        for col in df_insights.select_dtypes(include=['object']):
            df_insights[col] = df_insights[col].astype(str).str.replace('**', '', regex=False)
            
        cols = ["final_score", "type", "description"] + [c for c in df_insights.columns if c not in ["final_score", "type", "description"]]
        df_insights = df_insights[cols]
    st.dataframe(df_insights, use_container_width=True)
