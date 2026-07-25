import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Import shared helper functions from utils.py
from components.utils import (
    get_numeric_categorical,
    apply_iqr,
    iqr_report,
    make_summary
)
def render_eda_tab():
    st.subheader("📈 Visual Analytics & Dataset Health")
    
    if st.session_state.raw_df is None:
        st.info(" Please upload a CSV file in the sidebar to get started.")
        return

    df = st.session_state.raw_df

    # 1. Dataset Preview
    st.write("### Preview")
    st.dataframe(df.head(), use_container_width=True)

    # 2. Quick Metrics
    st.write("### Quick Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", len(df))
    with c2:
        st.metric("Columns", len(df.columns))
    with c3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))
    with c4:
        st.metric("Duplicate Rows", int(df.duplicated().sum()))

    # 3. Column Summary
    st.write("### Column Summary")
    st.dataframe(make_summary(df), use_container_width=True)

    # 4. Missing Values Chart
    st.write("### Missing Values by Column")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        missing.head(20).plot(kind="bar", ax=ax, color="#3B82F6")
        ax.set_title("Top Missing Columns")
        ax.set_ylabel("Missing Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # Free memory
    else:
        st.success("✨ No missing values found in this dataset.")

    # 5. Basic Interactive Charts
    st.write("### Basic Charts")
    numeric_cols, categorical_cols = get_numeric_categorical(df)

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Histogram", "Boxplot", "Countplot", "Scatterplot", "Correlation Heatmap"],
        key="eda_chart_type"
    )

    if chart_type == "Histogram" and numeric_cols:
        selected_col = st.selectbox("Select Numeric Column", numeric_cols, key="hist_col")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color="#2563EB")
        ax.set_title(f"Histogram of {selected_col}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif chart_type == "Boxplot" and numeric_cols:
        selected_col = st.selectbox("Select Numeric Column", numeric_cols, key="box_col")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, y=selected_col, ax=ax, color="#10B981")
        ax.set_title(f"Boxplot of {selected_col}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif chart_type == "Countplot" and categorical_cols:
        selected_col = st.selectbox("Select Categorical Column", categorical_cols, key="count_col")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x=selected_col, ax=ax, palette="viridis")
        ax.set_title(f"Countplot of {selected_col}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif chart_type == "Scatterplot" and len(numeric_cols) >= 2:
        x_col = st.selectbox("X-axis", numeric_cols, index=0, key="scatter_x")
        y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_y")
        
        if x_col != y_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="#F59E0B")
            ax.set_title(f"{x_col} vs {y_col}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Please choose two different numeric columns.")

    elif chart_type == "Correlation Heatmap" and len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("⚠️ Not enough suitable columns available for this chart type.")