import pandas as pd
import numpy as np
import streamlit as st

def get_numeric_categorical(df:pd.DataFrame) :
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols,categorical_cols

def make_summary(df:pd.DataFrame)-> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non Null": df.notnull().sum().values,
        "Null": df.isnull().sum().values,
        "Unique": df.nunique().values,
    })

def generate_df_context(df: pd.DataFrame) -> str:
    """Generates a detailed statistical string context of the dataset for Gemini."""
    num_cols, cat_cols = get_numeric_categorical(df)
    missing_summary = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    
    context = f"""
    Dataset Profile:
    - Shape: {df.shape[0]} rows, {df.shape[1]} columns
    - Numeric Columns ({len(num_cols)}): {num_cols}
    - Categorical Columns ({len(cat_cols)}): {cat_cols}
    - Total Missing Values: {df.isnull().sum().sum()}
    - Missing Breakdown: {missing_summary}
    - Duplicate Rows: {df.duplicated().sum()}

    Five-Row Sample Preview:
    {df.head(5).to_string()}

    Summary Statistics (Numeric):
    {df.describe().to_string() if num_cols else 'No numeric columns.'}
    """
    return context

def iqr_report(df:pd.DataFrame,columns):
    rows=[]
    for col in columns:
        s=pd.to_numeric(df[col],errors='coerce').dropna()
        if s.empty:
            continue
        q1=s.quantile(0.25)
        q3=s.quantile(0.75)
        iqr=q3-q1

        if iqr==0:
            outlier_count=0
            lower=upper=np.nan
        else:
            lower=q1-1.5*iqr
            upper=q3+1.5*iqr
            outlier_count=((
                pd.to_numeric(df[col], errors="coerce") < lower) |
                             (pd.to_numeric(df[col], errors="coerce") > upper)).sum()
        rows.append({
            "Column":col,
            "Q1":q1,
            "Q3":q3,
            "IQR":iqr,
            "Lower_Bound":lower,
            "Upper_bound":upper,
            "Outlier Rows":int(outlier_count),
        })
    return pd.DataFrame(rows)

def get_iqr_bounds(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None, None, None, None

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return q1, q3, iqr, None, None

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper

def apply_iqr(df:pd.DataFrame,columns,treatment:str):
    if not columns:
        return df.copy(),0
    treated_df=df.copy()

    affected=0

    for col in columns:

        s=pd.to_numeric(treated_df[col],errors='coerce')
        q1,q3,iqr,lower,upper=get_iqr_bounds(s)

        #skip if bounds are none
        if lower is None or upper is None:
            continue
        
        outlier_mask=(s<lower)|(s>upper)

        outlier_count=outlier_mask.sum()

        if outlier_count==0:
            st.success(f"✅ No outliers detected in {col}.")
            continue
        #capping or winsorizationation
        if treatment=="Cap(Winsorize)":
            treated_df.loc[s<lower,col]=lower
            treated_df.loc[s>upper,col]=upper
        
        elif treatment=="Replace with Median":
            median_value=s.median()
            treated_df.loc[outlier_mask,col]=median_value

        elif treatment=="Replace with Mean":
            mean_value=s.mean()
            treated_df.loc[outlier_mask, col] = mean_value
        elif treatment == "Remove Rows":

            treated_df = treated_df.loc[~outlier_mask]

        affected += outlier_count

    return treated_df, affected