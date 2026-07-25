# DataPilot AI 📊🤖

**DataPilot AI** is an end-to-end, interactive Data Science and Analytics workstation built with Python and Streamlit. It streamlines the entire data pipeline—from raw CSV uploading and automated Exploratory Data Analysis (EDA) to advanced Data Cleaning, Machine Learning modeling, and interactive AI-assisted insights powered by Google's Gemini LLM.

---

## 🌟 Key Features

### 1. 📈 Exploratory Data Analysis (EDA)
- **Dataset Overview**: Instant metrics on dataset dimensions, data types, missing value percentages, and key statistics.
- **Visual Explorations**: Interactive numeric and categorical distributions powered by Matplotlib and Seaborn.
- **Correlation Analysis**: Dynamic heatmaps to uncover underlying linear relationships between features.

### 2. 🧼 Preprocessing & Data Cleaning Pipeline
- **Column Dropping**: Interactively drop irrelevant or high-null columns.
- **Missing Value Imputation**: Flexible missing value handling using `sklearn.impute.SimpleImputer` for numerical (Mean/Median) and categorical (Most Frequent) attributes.
- **IQR Outlier Management**: Automatic IQR outlier detection with modular treatment methods including Winsorization (Capping), Mean/Median replacement, or row deletion.
- **Dataset Export**: Download processed, clean datasets with a single click.

### 3. 🤖 Machine Learning Studio
- **Supervised Learning**: Train Classification and Regression models on preprocessed data.
- **Evaluation Metrics**: Performance reports including Accuracy, Precision, Recall, F1-Score, R² Score, and RMSE.
- **Model Persistence**: Export trained models into serialized `.pkl` format using `pickle` for downstream deployment.

### 4. 💬 GenAI Dataset Assistant (Gemini API)
- **Context-Aware Analytics**: Generates real-time contextual summaries (`make_summary`) sent directly to Google Gemini LLM.
- **Quick Action Reports**: Instant insights on data quality issues, simplified executive summaries, and top feature drivers.
- **Interactive Querying**: Natural language query engine allowing users to ask bespoke questions about their dataset.

---

## 🛠️ Tech Stack & Libraries

- **Language**: Python 3.10+
- **Frontend Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: `pandas`, `numpy`
- **Data Preprocessing & ML**: `scikit-learn` (`SimpleImputer`, ML models)
- **Visualization**: `matplotlib`, `seaborn`
- **GenAI Integration**: `google-generativeai`
- **Model Serialization**: `pickle`

---

## 📁 Project Structure

```text
datapilot_ai/
│
├── .streamlit/
│   └── secrets.toml          # Local secret keys (API Keys) - Excluded from Git
├── components/
│   ├── __init__.py           # Package marker file
│   ├── tab_eda.py            # EDA UI and plotting components
│   ├── tab_clean.py          # Preprocessing and cleaning pipeline
│   ├── tab_ml.py             # Machine learning training & export
│   ├── tab_ai.py             # GenAI Chatbot interface
│   └── utils.py              # Helper functions (IQR, summary stats)
├── ai_engine.py              # Google Gemini API connector & query engine
├── my.py                     # Main Streamlit application entry point
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies for deployment
└── .gitignore                # Git exclusion rules
