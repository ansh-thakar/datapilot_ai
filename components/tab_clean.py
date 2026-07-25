import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from components.utils import (
    get_numeric_categorical,
    apply_iqr,
    iqr_report,
    make_summary
)

def render_clean_tab():
    st.subheader("🧼 Preprocessing Pipeline")
    
    if st.session_state.processed_df is None:
        st.info("Please upload a CSV file first.")
    else:
        clean_df = st.session_state.processed_df.copy()
        st.write("### Current Data Preview")
        st.dataframe(clean_df.head(), use_container_width=True)
        st.divider()

        # Phase 1: Drop Columns
        st.markdown("### 1) Drop Irrelevant Columns")
        cols_to_drop = st.multiselect(
            "Select columns to drop",
            clean_df.columns.tolist(),
            key="drop_cols"
        )
        if st.button("Execute Drop", key="drop_btn"):
            if cols_to_drop:
                clean_df = clean_df.drop(columns=cols_to_drop)
                st.session_state.processed_df = clean_df
                st.session_state.last_action = f"Dropped columns: {', '.join(cols_to_drop)}"
                st.success("Columns dropped successfully.")
                st.rerun()
            else:
                st.warning("Select at least one column to drop.")

        st.divider()

        # Phase 2: Missing Value Imputation
        st.markdown("### 2) Missing Value Imputation")
        num_cols, cat_cols = get_numeric_categorical(clean_df)
        col1, col2 = st.columns(2)

        with col1:
            num_strategy = st.selectbox(
                "Numeric strategy",
                ["None", "mean", "median"],
                key="num_strategy"
            )
        with col2:
            cat_strategy = st.selectbox(
                "Categorical strategy",
                ["None", "most_frequent"],
                key="cat_strategy"
            )

        if st.button("Apply Imputation", key="impute_btn"):
            working_df = clean_df.copy()
            changes_made = False

            if num_strategy != "None" and num_cols:
                num_imputer = SimpleImputer(strategy=num_strategy)
                working_df[num_cols] = num_imputer.fit_transform(working_df[num_cols])
                changes_made = True
            if cat_strategy != "None" and cat_cols:
                cat_imputer = SimpleImputer(strategy=cat_strategy)
                working_df[cat_cols] = cat_imputer.fit_transform(working_df[cat_cols])
                changes_made = True
            
            if changes_made:
                st.session_state.processed_df = working_df
                st.session_state.last_action = "Missing values imputed."
                st.success("Imputation completed successfully.")
                st.rerun()
            else:
                st.warning("Choose at least one imputation strategy.")

        st.divider()

        # Phase 3: IQR Outlier Detection / Treatment
        st.markdown("### 3) IQR Outlier Detection & Treatment")
        numeric_cols = st.session_state.processed_df.select_dtypes(include="number").columns.tolist()
        
        if not numeric_cols:
            st.info("No numeric columns found for outlier detection.")
        else:
            check_cols = st.multiselect(
                "Select numeric columns to inspect",
                numeric_cols,
                default=numeric_cols,
                key="outlier_cols"
            )
            
            if st.button("Detect Outliers (IQR)", key="detect_outliers_btn"):
                report = iqr_report(st.session_state.processed_df, check_cols)
                st.session_state.outlier_report = report
                if report.empty:
                    st.info("No numeric data available for the selected columns.")
                else:
                    total_outliers = int(report["Outlier Rows"].sum())
                    if total_outliers > 0:
                        st.warning("Outliers found in your data.")
                    else:
                        st.success("No outliers detected with the IQR method.")

        # Outlier Report Display & Treatment Actions
        if st.session_state.get("outlier_report") is not None and not st.session_state.outlier_report.empty:
            st.write("### Outlier Report")
            st.dataframe(st.session_state.outlier_report, use_container_width=True)

            total_outliers = int(st.session_state.outlier_report["Outlier Rows"].sum())
            if total_outliers > 0:
                treatment = st.radio(
                    "There are outliers in your data. How do you want to treat them?",
                    [
                        "Cap (Winsorize)",
                        "Replace with Median",
                        "Replace with Mean",
                        "Remove Rows",
                        "Keep as Is",
                    ],
                    horizontal=False,
                    key="outlier_treatment"
                )
                st.info("Recommended: Cap (Winsorize) because it keeps all rows and reduces extreme values.")
                
                if st.button("Apply Outlier Treatment", key="apply_outlier_treatment_btn"):
                    if treatment == "Keep as Is":
                        st.session_state.last_action = "Kept outliers unchanged."
                        st.success("No changes made to outliers.")
                    else:
                        # 🟢 Safely assign updated_df and affected
                        updated_df, affected = apply_iqr(
                            st.session_state.processed_df,
                            check_cols,
                            treatment
                        )
                        st.session_state.processed_df = updated_df
                        
                        if treatment == "Remove Rows":
                            st.session_state.last_action = f"Removed {affected} outlier rows."
                            st.success(f"Removed {affected} rows containing outliers.")
                        else:
                            st.session_state.last_action = f"Applied '{treatment}' to {affected} outlier values."
                            st.success(f"Applied '{treatment}' to {affected} outlier values.")
                    
                    st.rerun()

        st.divider()

        # Phase 4: Export Dataset
        st.markdown("### 4) Export Clean Dataset")
        csv_data = st.session_state.processed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned Dataset",
            data=csv_data,
            file_name="datapilot_ai_cleaned_output.csv",
            mime="text/csv"
        )

        st.write("### Current Cleaned Preview")
        st.dataframe(st.session_state.processed_df.head(), use_container_width=True)