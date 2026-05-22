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
def process_data(file_obj_or_path):
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
        df_raw, df_proc, profile, all_insights = process_data(uploaded_file)
    elif os.path.exists(data_path):
        df_raw, df_proc, profile, all_insights = process_data(data_path)
    else:
        st.info("Please upload a CSV or generate the sample data via terminal (`python data/generate_data.py`).")
        st.stop()

insights = [i for i in all_insights if i.get("final_score", 0) >= min_score][:num_insights_to_show]

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
        for x, y, v in sorted(strong_pos, key=lambda x: x[2], reverse=True):
            if not any((y == ux and x == uy) for ux, uy, _ in unique_pos):
                unique_pos.append((x, y, v))
                
        unique_neg = []
        for x, y, v in sorted(strong_neg, key=lambda x: x[2]):
            if not any((y == ux and x == uy) for ux, uy, _ in unique_neg):
                unique_neg.append((x, y, v))
                
        st.markdown("<p style='margin-top:15px; margin-bottom:5px; font-weight:bold; color:#fff;'>🔍 What this tells us about your active dataset:</p>", unsafe_allow_html=True)
        if unique_pos or unique_neg:
            for x, y, v in unique_pos[:2]:
                st.markdown(f"📈 **Direct Co-Movement**: **{clean_feature_name(x)}** and **{clean_feature_name(y)}** move in the same direction with a positive correlation of **{v:.2f}**. Elevating one will likely drive the other upward.")
            for x, y, v in unique_neg[:2]:
                st.markdown(f"📉 **Inverse Constraint**: **{clean_feature_name(x)}** and **{clean_feature_name(y)}** move in opposite directions with a negative correlation of **{v:.2f}**. Increases in one correspond to drops in the other.")
        else:
            st.markdown("ℹ️ No strong linear correlations (above ±0.45) detected in this dataset. The numeric fields seem to operate independently.")
        
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
    t = itype.lower()
    if 'correlation' in t: return 'badge-correlation'
    if 'cluster' in t: return 'badge-cluster'
    if 'association' in t: return 'badge-association'
    if 'lag' in t: return 'badge-timelag'
    if 'anomaly' in t: return 'badge-anomaly'
    return ''

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
    
    if top_sims:
        with st.expander("✨ Auto-Simulated Recommended Scenarios", expanded=True):
            cols_sim = st.columns(len(top_sims))
            for s_idx, ts in enumerate(top_sims):
                with cols_sim[s_idx]:
                    st.markdown(f"<p style='font-size:1.1rem; font-weight:bold; margin-bottom:5px; color:#fff;'>Scenario #{s_idx+1}</p>", unsafe_allow_html=True)
                    st.info(md_to_html(ts['narrative']))
                    st.markdown(md_to_html(ts['recommendation']))
                    st.markdown("---")
                
    st.divider()
    st.subheader("Custom Simulation Run")
    num_cols = profile["numeric_cols"]
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns to run simulations.")
    else:
        col_target, col_change, col_val = st.columns(3)
        target_col = col_target.selectbox("Target Outcome Predictor", num_cols, index=0, format_func=clean_feature_name)
        change_col = col_change.selectbox("Influential Variable to Modify", num_cols, index=1, format_func=clean_feature_name)
        pct_change = col_val.slider("% Intervention Change", -50, 100, 10)
        
        if st.button("🚀 Run AI Simulation"):
            with st.spinner("Running Random Forest Simulation..."):
                time.sleep(0.5) 
                res = sim.simulate(df_proc, target_col, change_col, pct_change)
                if "error" in res:
                    st.error(res["error"])
                else:
                    # Bold & clean callouts at the top
                    st.info(md_to_html(res["narrative"]))
                    st.markdown(md_to_html(res["recommendation"]))
                    
                    st.divider()
                    
                    metric_col, chart_col = st.columns([1, 1], gap="large")
                    with metric_col:
                        st.markdown(f"**Impact Summary:**")
                        clean_target = clean_feature_name(target_col)
                        st.metric(f"Simulated Shift in {clean_target}", f"{res['delta_pct']:.2f}%", f"{res['predicted_modified_mean'] - res['predicted_original_mean']:.2f} Abs Units")
                        
                        chart_data = pd.DataFrame({
                            "Scenario": ["Actual Baseline", "Simulated Intervention"],
                            "Predicted Mean": [res["predicted_original_mean"], res["predicted_modified_mean"]]
                        })
                        fig_bar = px.bar(chart_data, x="Scenario", y="Predicted Mean", color="Scenario", 
                                        color_discrete_sequence=['#888888', '#FF416C'], text_auto='.2f', template="plotly_dark")
                        fig_bar.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                    with chart_col:
                        st.markdown("🔑 **Key Features Influencing this Target Variable:**")
                        st.caption("These features are detected by the AI model as having the highest relative predictive strength.")
                        for driver in res["drivers"]:
                            st.write(f"**{driver['feature']}** ({driver['importance']:.1f}% relative strength)")
                            st.progress(driver['importance'] / 100.0)

with tab4:
    st.header("📈 Interactive Data Explorer")
    st.markdown("Visually explore the raw dataset by building your own charts automatically.")
    
    col_chart, col_x, col_y = st.columns(3)
    chart_type = col_chart.selectbox("Select Chart Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Histogram", "Pie Chart"])
    
    all_cols = profile["numeric_cols"] + profile["categorical_cols"]
    x_axis = col_x.selectbox("X-Axis Feature (Names/Categories)", all_cols, index=0)
    
    if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Pie Chart"]:
        y_axis = col_y.selectbox("Y-Axis Feature (Values)", profile["numeric_cols"], index=1 if len(profile["numeric_cols"])>1 else 0)
    else:
        y_axis = None
        
    color_col = st.selectbox("Color By (Optional Grouping)", ["None"] + profile["categorical_cols"])
    color_val = None if color_col == "None" else color_col
    
    custom_colors = ['#FF416C', '#00C9FF', '#92FE9D', '#FFB75E', '#8A2387', '#E94057']
    
    if chart_type == "Scatter Plot":
        fig = px.scatter(df_raw, x=x_axis, y=y_axis, color=color_val, template="plotly_dark", color_discrete_sequence=custom_colors)
        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    elif chart_type == "Line Chart":
        fig = px.line(df_raw, x=x_axis, y=y_axis, color=color_val, template="plotly_dark", color_discrete_sequence=custom_colors)
        fig.update_traces(line=dict(width=3))
    elif chart_type == "Bar Chart":
        fig = px.bar(df_raw, x=x_axis, y=y_axis, color=color_val, template="plotly_dark", color_discrete_sequence=custom_colors)
        fig.update_traces(marker_line_width=0)
    elif chart_type == "Box Plot":
        fig = px.box(df_raw, x=x_axis, y=y_axis, color=color_val, template="plotly_dark", color_discrete_sequence=custom_colors)
    elif chart_type == "Histogram":
        fig = px.histogram(df_raw, x=x_axis, color=color_val, template="plotly_dark", color_discrete_sequence=custom_colors)
        fig.update_traces(opacity=0.8, marker_line_width=1, marker_line_color='black')
    elif chart_type == "Pie Chart":
        fig = px.pie(df_raw, names=x_axis, values=y_axis, color=color_val, template="plotly_dark", color_discrete_sequence=custom_colors, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#1E1E24', width=2)))
        
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, l=20, r=20, b=20),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=False),
        font=dict(color='#e0e0e0'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
        
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("⚙️ Raw Pipeline Details")
    st.write("Browse through all discovered vectors beneath the hood.")
    df_insights = pd.DataFrame(all_insights)
    if not df_insights.empty:
        cols = ["final_score", "type", "description"] + [c for c in df_insights.columns if c not in ["final_score", "type", "description"]]
        df_insights = df_insights[cols]
    st.dataframe(df_insights, use_container_width=True)
