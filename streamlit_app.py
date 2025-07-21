# todo: drop ID column from train and test data
# todo: refine EDA section
# todo: refine SHAP section
# todo: refine missing value imputation and feature engineering

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Set page config for better layout
st.set_page_config(layout="wide", page_title="Data Science Project Demo")

# Global Variables and Constants
TARGET_COLUMN = "Personality"
ID_COLUMN = "id"
MAPPING_YES_NO = {
    "Yes": 1,
    "No": 0,
}
MAPPING_PERSONALITY = {
    "Introvert": 0,
    "Extrovert": 1,
}
MAPPING_PERSONALITY_INV = {
    0: "Introvert",
    1: "Extrovert",
}
PARAMS = {
    "colsample_bytree": 1.0,
    "learning_rate": 0.2,
    "max_depth": 3,
    "n_estimators": 200,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "subsample": 0.8,
}


# Data Loading
@st.cache_data  # Cache data loading to avoid re-running on every interaction
def load_data(data_path):
    """Loads a dataset."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Make sure '{data_path}' exists in the 'data' directory. ")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()


def filter_data(df, target_column=TARGET_COLUMN, id_column=ID_COLUMN):
    """
    Filters the dataframe to remove the target and ID columns if they exist.
    This is useful for preprocessing steps where these columns should not be included.
    """
    id_col = df.get(id_column, None)
    if target_column in df.columns:
        df = df.drop(columns=[target_column], errors="ignore")
    if id_column in df.columns:
        df = df.drop(columns=[id_column], errors="ignore")
    return df, id_col


# Data Preprocessing - Encoding
def map_data(df, mapping_dict):
    """
    Maps values in the dataframe based on a provided mapping dictionary.
    This is useful for converting categorical string labels to numeric values.
    """
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


# Data Preprocessing - Imputation
def create_imputer(df, numerical_cols=None, categorical_cols=None):
    """
    Creates and fits a preprocessing pipeline based on the training data.
    Handles numerical and categorical features imputation.
    """
    numerical_imputer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_imputer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # Create a column transformer to apply different transformations to different columns
    imputer = ColumnTransformer(
        transformers=[
            ("num", numerical_imputer, numerical_cols),
            ("cat", categorical_imputer, categorical_cols),
        ],
        remainder="passthrough",
    )
    return imputer


def impute_data(df, preprocessor, fit_preprocessor=False):
    """Applies the imputation pipeline to the dataframe."""
    if fit_preprocessor:
        # Fit and transform (for training data)
        X_imputed = preprocessor.fit_transform(df)
    else:
        # Transform only (for test or new prediction data)
        X_imputed = preprocessor.transform(df)
    return X_imputed


# Model Training
def train_model(X_train_processed, y_train, params={}):
    """Trains an XGBoost Classifier model with optional hyperparameters."""
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": accuracy_score,
        "n_estimators": 100,
        "random_state": 42,
    }
    # Update default params with any provided kwargs
    default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train_processed, y_train)
    return model


# Streamlit UI
def main():
    st.title("🚀 Data Science Project Showcase")
    st.markdown("""
        This project demonstrates a comprehensive data science workflow, from data preprocessing and modeling
        to containerization and interactive visualization.
    """)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Project Introduction",
            "Data Overview",
            "Model Performance & Insights",
            "Batch Prediction",
        ],
    )

    df = load_data("data/train.csv")

    # Ensure the target column exists in the training data
    if TARGET_COLUMN not in df.columns:
        st.error(
            f"Error: The target column '{TARGET_COLUMN}' was not found in 'train.csv'. Please adjust the TARGET_COLUMN variable in the code."
        )
        st.stop()

    # Split the data into features and target
    X = filter_data(df, TARGET_COLUMN, ID_COLUMN)[0]
    y = df[TARGET_COLUMN].map(MAPPING_PERSONALITY)

    # Identify numerical and categorical features
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply the mapping to the categorical column
    X_train_processed = map_data(
        X_train,
        {cat_col: MAPPING_YES_NO for cat_col in categorical_cols},
    )
    X_test_processed = map_data(
        X_test,
        {cat_col: MAPPING_YES_NO for cat_col in categorical_cols},
    )
    X_processed = map_data(
        X,
        {cat_col: MAPPING_YES_NO for cat_col in categorical_cols},
    )

    # Create and fit the preprocessor
    part_imputer = create_imputer(X_train_processed, numerical_cols, categorical_cols)
    X_train_imputed = impute_data(
        X_train_processed, part_imputer, fit_preprocessor=True
    )
    X_test_imputed = impute_data(X_test_processed, part_imputer, fit_preprocessor=False)

    imputer = create_imputer(X_processed, numerical_cols, categorical_cols)
    X_imputed = impute_data(X_processed, imputer, fit_preprocessor=True)

    # Train the model
    part_model = train_model(
        X_train_imputed,
        y_train,
        PARAMS,
    )
    model = train_model(
        X_imputed,
        y,
        PARAMS,
    )

    if page == "Project Introduction":
        st.header("🌟 Welcome!")
        st.markdown("""
            This interactive application serves as a demonstration of a complete data science pipeline.
            It covers:
            - **Data Preprocessing**: Handling raw data, cleaning, and transforming it into a suitable format for modeling.
            - **Machine Learning Modeling**: Training an **XGBoost Classifier** to make predictions.
            - **Model Interpretability**: Understanding *why* the model makes certain predictions using **SHAP values**.
            - **Reproducibility**: The entire setup is containerized using **Docker** for easy deployment.
            - **Interactive Interface**: This Streamlit app allows you to explore data, model performance, and even upload new data for batch predictions.

            Use the sidebar to navigate through different sections of the project.
        """)

    elif page == "Data Overview":
        st.header("📊 Dataset Head")
        st.markdown("Here's a glimpse of the raw datasets")

        st.subheader("Data (`train.csv`)")
        st.write(df.head())
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.write("---")
        st.subheader("Data Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.markdown("---")
        st.subheader("Missing Values Overview")
        st.write("Percentage of missing values in the data:")
        missing_train = df.isnull().sum() / len(df) * 100
        st.dataframe(
            missing_train[missing_train > 0]
            .sort_values(ascending=False)
            .to_frame(name="Missing %")
        )

    elif page == "Model Performance & Insights":
        st.header("📈 Model Performance & Feature Importance")
        st.markdown(
            "This section provides insights into the trained model's performance and explains its predictions using SHAP values. Noted that in this section, the model is trained on 80% of the data, and the remaining 20% is used for evaluation."
        )

        # Make predictions on the training data for evaluation (or a held-out validation set if available)
        # For simplicity in this demo, we'll evaluate on training data, but in a real scenario,
        # you'd use a separate validation/test set.
        y_pred_proba = part_model.predict_proba(X_test_imputed)[:, 1]
        y_pred = part_model.predict(X_test_imputed)

        st.subheader("ROC AUC Curve")
        st.markdown(
            "The Receiver Operating Characteristic (ROC) curve illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied."
        )

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax_roc.set_xlim(0.0, 1.0)
        ax_roc.set_ylim(0.0, 1.05)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        st.subheader("Feature Importance (SHAP Values)")
        st.markdown("""
            SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.
            It connects optimal credit allocation with local explanations using the classic Shapley values from game theory.
            The plot below shows the impact of each feature on the model's output.
        """)

        # Create a SHAP explainer and calculate SHAP values
        # For tree models, TreeExplainer is efficient
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_imputed)

        # SHAP Summary Plot
        fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_test_imputed,
            feature_names=numerical_cols + categorical_cols,
            show=False,
        )
        st.pyplot(fig_shap)
        st.markdown("---")
        st.markdown("""
            **Interpretation of SHAP Summary Plot:**
            - **X-axis**: SHAP value (impact on model output).
            - **Y-axis**: Features, ordered by importance.
            - **Color**: Feature value (red = high, blue = low).
            - Each dot represents an instance from the dataset.
            - Dots stacked vertically indicate density.
            - A dot to the right of zero means that feature value increases the prediction, and to the left decreases it.
        """)

    elif page == "Batch Prediction":
        st.header("📦 Batch Prediction")
        st.markdown("""
            Upload a CSV file containing new data (without the target column) for batch prediction.
            The application will preprocess the data, make predictions, and allow you to download the results.
        """)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                # Read the uploaded file
                new_data_df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data Head")
                st.write(new_data_df.head())

                # Apply the same preprocessing steps
                # Ensure the new data has the same columns as the training data, even if some are missing
                # We need to align columns before preprocessing
                # Get all columns from X_train (features used for training)
                whole_feature_cols = df.columns.tolist()

                # Add any missing columns to new_data_df and fill with NaN
                for col in whole_feature_cols:
                    if col not in new_data_df.columns:
                        new_data_df[col] = np.nan

                # Drop any extra columns in new_data_df that were not in training features
                new_data_df = new_data_df[whole_feature_cols]

                # Separate features and target
                X_new, id_new = filter_data(new_data_df, TARGET_COLUMN, ID_COLUMN)

                # Apply the mapping to the categorical column
                X_new_processed = map_data(
                    X_new,
                    {cat_col: MAPPING_YES_NO for cat_col in categorical_cols},
                )

                # Create and fit the preprocessor
                X_new_imputed = impute_data(
                    X_new_processed, imputer, fit_preprocessor=False
                )

                # Make predictions
                predictions_encoded = model.predict(X_new_imputed)

                df_predictions = pd.DataFrame(
                    {
                        ID_COLUMN: id_new,
                        TARGET_COLUMN: predictions_encoded,
                    }
                )
                df_predictions[TARGET_COLUMN] = df_predictions[TARGET_COLUMN].map(
                    MAPPING_PERSONALITY_INV
                )

                st.subheader("Prediction Results (Head)")
                st.write(df_predictions.head())

                # Provide download link
                csv_output = df_predictions.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_output,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
                st.success("Predictions generated successfully!")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info(
                    "Please ensure your uploaded CSV has a similar structure to the training data."
                )


if __name__ == "__main__":
    main()
