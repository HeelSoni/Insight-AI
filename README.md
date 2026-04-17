# ✨ Insight AI
### Autonomous Data Discovery & Reasoning System

**Insight AI** is a production-ready, autonomous data analysis engine and beautiful dashboard that acts as an automated junior data scientist. It ingests raw datasets, discovers deep mathematical patterns without any predefined target variable, generates human-readable hypotheses, formulates follow-up questions, and runs counterfactual machine-learning simulations.

Built with **Python**, **Streamlit**, **Scikit-Learn**, and **Plotly** for a stunning dark-mode web experience.

---

## 🚀 Features

*   **⚡ Autonomous Discovery:** Automatically cleans, imputes, and encodes data to prepare it for deep mathematical analysis.
*   **🧠 Multi-Engine Architecture:** Detects Pearsons/Spearmans correlations, K-Means Clustering, Apriori Association Rules, and Time-Lag shifts.
*   **💬 Human-Readable Hypotheses:** Translates abstract mathematical findings into understandable business logic and follow-up curiosity questions.
*   **🎛️ Counterfactual Simulator:** Uses `RandomForestRegressor` to simulate "What-If" scenarios. For example: *"If I increase Salary by 10%, how much will Demand Score shift?"*
*   **📈 Interactive Data Explorer:** Dynamically build beautiful custom charts (Scatter, Line, Bar, Boxplot, Pie) using a neon Plotly UI.
*   **🎨 Stunning Cyber UI:** Implemented purely in Streamlit with custom CSS and interactive, transparent badges.

---

## 🛠️ Installation & Setup

You will need Python 3.9+ installed on your system.

### 1. Clone the repository
```bash
git clone https://github.com/HeelSoni/Insight-AI.git
cd Insight-AI/insight_ai
```

### 2. Create and Activate a Virtual Environment
**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure
```text
insight_ai/
│
├── app/
│   └── streamlit_app.py      # The Streamlit Dashboard UI
│
├── src/
│   ├── preprocessing.py      # Automated Data Cleaning & Encoding
│   ├── pattern_engine.py     # Math & ML Engines (KMeans, Apriori, Pearson)
│   ├── hypothesis.py         # Natural Language Generator 
│   ├── question_gen.py       # Anomaly detection & Question Generator
│   ├── simulator.py          # Random Forest Counterfactual Simulator
│   └── scoring.py            # Insight Ranking Logic (Novelty/Impact)
│
├── data/
│   ├── generate_data.py      # Script to generate a realistic local dataset
│   └── sample_sales_weather.csv 
│
├── requirements.txt          # Python module dependencies
└── README.md
```

---

## 💡 Usage Example
Once the dashboard is running, simply drag and drop any numeric/categorical `.csv` dataset into the sidebar. Use the top navigation tabs to browse auto-generated insights, simulate targeted interventions, or build interactive visualization plots!

---
*Developed by Heel Soni*
