from components.utils import (
    get_numeric_categorical,
    apply_iqr,
    iqr_report,
    make_summary
)
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error,
    confusion_matrix,
)
def render_ml_tab():
        st.subheader("🤖 Machine Learning Model Studio")
    
        if st.session_state.processed_df is None:
            st.info("Please upload and preprocess a dataset first.")
        else:
            df = st.session_state.processed_df.copy()
    
            st.write("### Cleaned Dataset Preview")
            st.dataframe(df.head(), width="stretch")
            all_columns=df.columns.tolist()
            if len(all_columns)<2:
                st.warning("Need atleast 2 columns for ML!")
            else:
                
                target_var = st.selectbox(
                "1️⃣ What outcome do you want to predict?",
                all_columns,
                help="Select the column you care about most (e.g., 'Target' or 'Salary' or 'Purchased')."
                )
    
                feature_vars = st.multiselect(
                "2️⃣ Which factors influence this outcome?",
                [c for c in all_columns if c != target_var],
                help="Choose columns like 'Experience', 'Age', or 'Education' that help explain the outcome."
                 )
            col_a, col_b = st.columns(2)
            with col_a:
                    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="test_size")
            with col_b:
                    model_type = st.selectbox(
                        "Choose Model",
                        [
                            "Logistic Regression",
                            "Linear Regression",
                            "Random Forest Classifier",
                            "Random Forest Regressor",
                        ],
                        key="model_type"
                    )
            
            scale_numeric=st.checkbox(
                    "Standardize numeric features",
                    value=(model_type in ["Logistic Regression", "Linear Regression"]),
                    key="scale_numeric"
                )
    
            st.caption("Encoding is hidden. Imputation and scaling happen inside the training pipeline.")
            
            if st.button("🚀 Train Model", key="train_model_btn"):
                st.spinner("Building pipeline and training model...")
                if not feature_vars:
                    st.warning("Please select at least one feature.")
                else:
                    X = df[feature_vars].copy()
                    y = df[target_var].copy()
                    is_classification = model_type in ["Logistic Regression", "Random Forest Classifier"]
                    is_regression = model_type in ["Linear Regression", "Random Forest Regressor"]
    
                        # Basic target validation
                    if is_regression and not pd.api.types.is_numeric_dtype(y):
                        st.error("Regression requires a numeric target column.")
                    else:
                        # Encode target for classification if needed
                        label_encoder = None
                    if is_classification and not pd.api.types.is_numeric_dtype(y):
                        label_encoder = LabelEncoder()
                        y = label_encoder.fit_transform(y.astype(str))
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
                    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()
                                
                    numeric_steps=[("imputer",SimpleImputer(strategy='mean'))]
                    if scale_numeric:
                        numeric_steps.append(("scaler", StandardScaler()))
                        numeric_pipeline = Pipeline(steps=numeric_steps)
    
                    categorical_pipeline = Pipeline(steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown="ignore"))
                    ])
                    transformers = []
                    if numeric_features:
                        transformers.append(("num", numeric_pipeline, numeric_features))
                    if categorical_features:
                        transformers.append(("cat", categorical_pipeline, categorical_features))
    
                    preprocessor = ColumnTransformer(
                                transformers=transformers,
                                remainder="drop"
                    )
    
                    if model_type == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    elif model_type == "Linear Regression":
                        model = LinearRegression()
                    elif model_type == "Random Forest Classifier":
                        model = RandomForestClassifier(random_state=42)
                    else:
                        model = RandomForestRegressor(random_state=42)
                            
                    pipeline = Pipeline(steps=[
                                ("preprocessor", preprocessor),
                                ("model", model)
                    ])
                    st.spinner("Traning Model ......")
                    try:
                        pipeline.fit(X_train, y_train)
                        preds = pipeline.predict(X_test)
    
                        if is_classification:
                            acc = accuracy_score(y_test, preds)
                            st.success(f"Classification Accuracy: {acc:.2%}")
    
                            if label_encoder is not None:
                                st.caption("Target labels were encoded automatically for training.")
    
                        elif is_regression:
                            r2 = r2_score(y_test, preds)
                            mse = mean_squared_error(y_test, preds)
                            rmse = np.sqrt(mse)
                            st.success(f"R² Score: {r2:.3f}")
                            st.write(f"MSE: {mse:.3f}")
                            st.write(f"RMSE: {rmse:.3f}")
    
                            st.success("Model pipeline trained successfully.")
                            if model is not None:
                               st.success("Model trained successfully!")

                       # 1. Save model directly to a file on your disk
                               file_path = "trained_model.pkl"
                               with open(file_path, "wb") as f:
                                  pkl.dump(model, f)

                        #2. Read the raw bytes from the file
                               with open(file_path, "rb") as f:
                                   model_bytes = f.read()

                        # 3. Streamlit download button using raw bytes
                               st.download_button(
                              label="📦 Download Trained Model (.pkl)",
                              data=model_bytes,
                              file_name="datapilot_trained_model.pkl",
                              mime="application/octet-stream",
                               key="download_model_btn"
                                )
                    except Exception as e:
                            st.error(f"Training failed: {e}")

    

