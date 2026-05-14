<div align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/2103/2103832.png" alt="Logo" width="80" height="80">

  <h1 align="center">✨ Insight AI</h1>
  <p align="center">
    <strong>Autonomous Data Discovery & Reasoning System</strong>
    <br />
    An automated junior data scientist built with Streamlit and Scikit-Learn.
  </p>
  
  <p align="center">
    <a href="https://github.com/HeelSoni/Insight-AI/stargazers"><img src="https://img.shields.io/github/stars/HeelSoni/Insight-AI?style=for-the-badge&color=yellow" alt="Stars"></a>
    <a href="https://github.com/HeelSoni/Insight-AI/network/members"><img src="https://img.shields.io/github/forks/HeelSoni/Insight-AI?style=for-the-badge&color=blue" alt="Forks"></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  </p>
</div>

---

## 📖 Overview

**Insight AI** is a production-ready, autonomous data analysis engine and beautiful dashboard that acts as an automated data scientist. It is designed to ingest raw datasets, discover deep mathematical patterns without any predefined target variable, generate human-readable hypotheses, formulate follow-up questions, and run counterfactual machine-learning simulations.

Whether you are a business analyst looking for hidden trends, or a researcher needing quick exploratory data analysis (EDA), Insight AI automates the heavy lifting and presents findings in a stunning, neon-accented dark-mode web experience.

---

## 🚀 Core Features

### ⚡ 1. Autonomous Data Processing Pipeline
- **Auto-Cleaning:** Dynamically handles missing values using median imputation for numerics and mode imputation for categoricals.
- **Auto-Encoding:** Detects categorical columns and applies label encoding automatically.
- **Scaling & Normalization:** Standardizes numeric variance to prepare data for clustering and ML models.

### 🧠 2. Multi-Engine Pattern Discovery
- **Correlation Engines:** Analyzes Pearson and Spearman matrices to find strong linear and monotonic relationships.
- **Clustering Engine:** Uses **K-Means** to automatically group data points and identify hidden segment characteristics.
- **Association Rules:** Leverages the **Apriori** algorithm to find co-occurrences and transactional relationships (e.g., *if X is high, Y is also high*).
- **Time-Lag Analysis:** Evaluates historical shifts and lagging indicators if a temporal dimension is detected.

### 💬 3. Natural Language Hypothesis Generation
- Translates abstract mathematical findings (like `corr = 0.85`) into understandable business logic and hypotheses.
- Formulates autonomous follow-up curiosity questions (e.g., *"Why does group A exhibit behavior B?"*) based on anomaly detection.

### 🎛️ 4. Counterfactual ML Simulator ("What-If" Analysis)
- Automatically trains a `RandomForestRegressor` on your dataset on-the-fly.
- Allows users to adjust input sliders to simulate realistic interventions.
- Calculates and displays the projected shift in a target variable. For example: *"If I increase Salary by 10%, how much will the Demand Score shift?"*

### 📈 5. Interactive Data Explorer
- Dynamically build beautiful custom charts using a neon Plotly UI.
- Supports **Scatter, Line, Bar, Boxplot, and Pie Charts**.
- Fully interactive: zoom, pan, hover tooltips, and dynamic color-coding.

---

## 💻 Tech Stack & Architecture

Insight AI is built entirely in Python, utilizing modern data science and frontend frameworks:

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | `Streamlit` | Rapid, interactive web dashboard creation. |
| **Data Manipulation** | `Pandas`, `NumPy` | High-performance dataframe operations and numerical processing. |
| **Machine Learning** | `Scikit-Learn` | Clustering (K-Means), Regression (Random Forest), and Preprocessing. |
| **Pattern Mining** | `mlxtend` | Association rule mining and Apriori algorithms. |
| **Visualizations** | `Plotly`, `Seaborn`, `Matplotlib` | Interactive charts, neon-styled graphics, and statistical plotting. |

---

## 🛠️ Installation & Local Setup

You will need **Python 3.9+** installed on your system.

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

## ☁️ Deployment

Insight AI is deployment-ready.

**Deploying to Railway / Render:**
The repository includes a `Procfile` configured for deployment on platforms like Railway or Heroku.
Simply connect your GitHub repo to Railway, and it will automatically build and serve the Streamlit app.

**Deploying to Streamlit Community Cloud (Recommended):**
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Connect this repository.
3. Set the Main file path to `app/streamlit_app.py`.
4. Deploy for free!

---

## 📁 Project Architecture

```text
insight_ai/
│
├── app/
│   └── streamlit_app.py      # The main Streamlit Dashboard UI & routing
│
├── src/
│   ├── preprocessing.py      # Automated Data Cleaning, Imputation, & Encoding
│   ├── pattern_engine.py     # Math & ML Engines (KMeans, Apriori, Pearson)
│   ├── hypothesis.py         # Natural Language Generator for human-readable insights
│   ├── question_gen.py       # Anomaly detection & Question Generator
│   ├── simulator.py          # Random Forest Counterfactual Simulator logic
│   └── scoring.py            # Insight Ranking Logic (Novelty/Impact thresholds)
│
├── data/
│   ├── generate_data.py      # Script to generate realistic mock datasets
│   └── sample_sales_weather.csv 
│
├── Procfile                  # Production start command for PaaS deployment
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 💡 Usage Guide

1. **Upload Data:** Once the dashboard is running, drag and drop any numeric/categorical `.csv` dataset into the sidebar.
2. **Review Insights:** Navigate to the "Insights" tab to read the auto-generated findings and mathematical patterns.
3. **Simulate Scenarios:** Go to the "Simulator" tab, select a target variable, and move the sliders to see how changing features impacts the outcome.
4. **Explore Visually:** Use the "Data Explorer" tab to build custom, interactive Plotly charts.

---

<div align="center">
  <p><i>Developed by <b>Heel Soni</b></i></p>
</div>
