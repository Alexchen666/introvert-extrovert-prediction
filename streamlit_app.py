# todo: refine text

from typing import Dict, Optional, Tuple

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
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Set page config for better layout
st.set_page_config(
    layout="wide", page_title="Predict the Introverts from the Extroverts"
)

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
def load_data(data_path: str) -> pd.DataFrame:
    """Loads a dataset.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        df (pd.DataFrame): Loaded dataframe."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Make sure '{data_path}' exists in the directory. ")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()


def filter_data(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN, id_column: str = ID_COLUMN
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Filters the dataframe to remove the target and ID columns if they exist.

    Args:
        df (pd.DataFrame): Input dataframe to filter.
        target_column (str, optional): Name of the target column to remove. Defaults to TARGET_COLUMN.
        id_column (str, optional): Name of the ID column to remove. Defaults to ID_COLUMN.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.Series]]: A tuple containing:
            - df: Filtered dataframe with target and ID columns removed
            - id_col: ID column data if it existed, None otherwise
    """
    id_col = df.get(id_column, None)
    if target_column in df.columns:
        df = df.drop(columns=[target_column], errors="ignore")
    if id_column in df.columns:
        df = df.drop(columns=[id_column], errors="ignore")
    return df, id_col


# Data Preprocessing - Encoding
def map_data(df: pd.DataFrame, mapping_dict: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """
    Maps values in the dataframe based on a provided mapping dictionary.

    Args:
        df (pd.DataFrame): Input dataframe to apply mappings to.
        mapping_dict (Dict[str, Dict[str, int]]): Dictionary where keys are column names
            and values are dictionaries mapping old values to new values.

    Returns:
        df (pd.DataFrame): Dataframe with mapped values applied to specified columns.
    """
    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


# Data Preprocessing - Imputation
def create_imputer(
    numerical_cols: Optional[list] = None,
    categorical_cols: Optional[list] = None,
) -> ColumnTransformer:
    """
    Creates and fits a preprocessing pipeline based on the training data.
    Handles numerical and categorical features imputation.

    Args:
        numerical_cols (Optional[list], optional): List of numerical column names. Defaults to None.
        categorical_cols (Optional[list], optional): List of categorical column names. Defaults to None.

    Returns:
        imputer (ColumnTransformer): Configured column transformer for imputation.
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
        sparse_threshold=0,
    )
    return imputer


def impute_data(
    df: pd.DataFrame, preprocessor: ColumnTransformer, fit_preprocessor: bool = False
) -> np.ndarray:
    """
    Applies the imputation pipeline to the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe to apply imputation to.
        preprocessor (ColumnTransformer): Column transformer for imputation.
        fit_preprocessor (bool, optional): Whether to fit the preprocessor. Defaults to False.

    Returns:
        np.ndarray: Imputed data as numpy array.
    """
    if fit_preprocessor:
        # Fit and transform (for training data)
        X_imputed = preprocessor.fit_transform(df)
    else:
        # Transform only (for test or new prediction data)
        X_imputed = preprocessor.transform(df)

    return X_imputed  # type: ignore


# Model Training
def train_model(
    X_train_processed: np.ndarray,
    y_train: pd.Series,
    params: dict = {},
) -> xgb.XGBClassifier:
    """
    Trains an XGBoost Classifier model with optional hyperparameters.

    Args:
        X_train_processed (np.ndarray): Preprocessed training features.
        y_train (pd.Series): Training target values.
        params (dict, optional): Hyperparameters for the model. Defaults to {}.

    Returns:
        model (xgb.XGBClassifier): Trained XGBoost classifier model.
    """
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
def main() -> None:
    """
    Main function that runs the Streamlit application.
    """
    st.title("🚀 Predict the Introverts from the Extroverts")
    st.markdown("""
        This project demonstrates a comprehensive data science workflow, from data preprocessing and modelling
        to interactive visualisation.
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
    part_imputer = create_imputer(numerical_cols, categorical_cols)
    X_train_imputed = impute_data(
        X_train_processed, part_imputer, fit_preprocessor=True
    )
    X_test_imputed = impute_data(X_test_processed, part_imputer, fit_preprocessor=False)

    imputer = create_imputer(numerical_cols, categorical_cols)
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
            The work of Kaggle competition - [Predict the Introverts from the Extroverts Playground Series - Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7/overview).
        
            This interactive application serves as a demonstration of a complete data science pipeline.
            It covers:
            - Data Preprocessing: Handling raw data, cleaning, and transforming it into a suitable format for modeling.
            - Machine Learning Modelling: Training an XGBoost Classifier to make predictions.
            - Model Interpretability: Understanding why the model makes certain predictions using SHAP values.
            - Interactive Interface: This Streamlit app allows you to explore data, model performance, and even upload new data for batch predictions.

            Use the sidebar to navigate through different sections of the project.
        """)
        st.subheader("Project Objective")
        st.markdown(
            "Predict whether a person is an introvert or extrovert, given their social behavior and personality traits."
        )

    elif page == "Data Overview":
        st.header("📊 Exploratory Data Analysis")

        # Data Shape
        st.subheader("Dataset Shape")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Features", df.shape[1] - 2)  # Excluding target and id column

        st.markdown("""
            Features:
            - id: Unique identifier for each individual.
            - Time_spent_Alone: Hours spent alone daily (0-11).
            - Stage_fear: Presence of stage fright (Yes/No).
            - Social_event_attendance: Frequency of social events (0-10).
            - Going_outside: Frequency of going outside (0-7).
            - Drained_after_socializing: Feeling drained after socializing (Yes/No).
            - Friends_circle_size: Number of close friends (0-15).
            - Post_frequency: Social media post frequency (0-10).
            - Personality: Target variable (Extrovert/Introvert).
        """)

        # Dataset Head
        st.subheader("Dataset Head")
        st.dataframe(df.head(10))

        st.markdown("---")

        # Column Information with Missing Values
        st.subheader("Column Information & Missing Values")

        # Create comprehensive column info table
        column_info = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            column_info.append(
                {
                    "Column Name": col,
                    "Data Type": str(df[col].dtype),
                    "Missing Count": missing_count,
                    "Missing %": f"{missing_pct:.2f}%",
                    "Unique Values": df[col].nunique(),
                }
            )

        column_info_df = pd.DataFrame(column_info)
        st.dataframe(column_info_df, use_container_width=True)

        st.markdown("---")

        # Descriptive Statistics
        st.subheader("Descriptive Statistics")

        # Overall descriptive statistics
        desc_stats = df.describe().T
        desc_stats = desc_stats.round(2)
        st.dataframe(desc_stats, use_container_width=True)

        st.markdown("---")

        # Variable Distribution Analysis
        st.subheader("Variable Distribution Analysis")

        # Combine all variables for selection
        all_variables = []
        for col in numerical_cols:
            all_variables.append(f"{col} (Numerical)")
        for col in categorical_cols:
            all_variables.append(f"{col} (Categorical)")

        # Add target column
        if df[TARGET_COLUMN].dtype in [
            "int64",
            "float64",
        ]:
            all_variables.append(f"{TARGET_COLUMN} (Numerical)")
        else:
            all_variables.append(f"{TARGET_COLUMN} (Categorical)")

        if all_variables:
            selected_variable = st.selectbox(
                "Select a variable to visualise its distribution:",
                all_variables,
                key="variable_select",
            )

            # Extract variable name and type
            if selected_variable:
                var_name = selected_variable.split(" (")[0]
                var_type = (
                    "Numerical" if "(Numerical)" in selected_variable else "Categorical"
                )
            else:
                var_name = ""
                var_type = ""

            if var_type == "Numerical":
                col1, col2 = st.columns(2)

                with col1:
                    # Histogram
                    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
                    sns.histplot(
                        data=df, x=var_name, ax=ax_hist, kde=False, color="skyblue"
                    )
                    ax_hist.set_title(f"Distribution of {var_name}")
                    ax_hist.set_xlabel(var_name)
                    st.pyplot(fig_hist)

                with col2:
                    # Box plot
                    fig_box, ax_box = plt.subplots(figsize=(8, 6))
                    sns.boxplot(
                        data=df,
                        x=var_name,
                        ax=ax_box,
                        color="lightgreen",
                    )
                    ax_box.set_title(f"Box Plot of {var_name}")
                    st.pyplot(fig_box)

                # Summary statistics for selected variable
                st.write(f"**Summary Statistics for {var_name}:**")
                stats_df = df[var_name].describe().to_frame().T
                st.dataframe(stats_df, use_container_width=True)

            else:  # Categorical
                col1, col2 = st.columns(2)

                with col1:
                    # Bar plot
                    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
                    sns.countplot(
                        data=df,
                        x=var_name,
                        ax=ax_bar,
                        order=df[var_name].value_counts().index,
                        palette="pastel",
                    )
                    ax_bar.set_title(f"Distribution of {var_name}")
                    ax_bar.set_xlabel(var_name)
                    ax_bar.set_ylabel("Count")
                    ax_bar.tick_params(axis="x", rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_bar)

                with col2:
                    # Value counts table
                    st.write(f"**Value Counts for {var_name}:**")
                    counts_df = df[var_name].value_counts().reset_index()
                    st.dataframe(counts_df, use_container_width=True)
        else:
            st.info("No variables available for visualisation.")

        st.markdown("---")

        # Correlation Heatmap
        st.subheader("Correlation Analysis")

        if len(numerical_cols) > 1:
            # Include target if it's numerical
            corr_cols = numerical_cols.copy()
            if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dtype in [
                "int64",
                "float64",
            ]:
                corr_cols.append(TARGET_COLUMN)

            correlation_matrix = df[corr_cols].corr()

            fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix,
                mask=mask,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                ax=ax_corr,
                fmt=".2f",
            )
            ax_corr.set_title("Correlation Heatmap of Numerical Variables")
            plt.tight_layout()
            st.pyplot(fig_corr)

            # Show strongest correlations
            st.write("**Strongest Correlations (excluding self-correlations):**")
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_pairs.append(
                        {
                            "Variable 1": correlation_matrix.columns[i],
                            "Variable 2": correlation_matrix.columns[j],
                            "Correlation": correlation_matrix.iloc[i, j],
                        }
                    )

            corr_pairs_df = pd.DataFrame(corr_pairs)
            corr_pairs_df = corr_pairs_df.reindex(
                corr_pairs_df["Correlation"].abs().sort_values(ascending=False).index
            )
            st.dataframe(corr_pairs_df.head(10), use_container_width=True)
        else:
            st.info("Need at least 2 numerical variables to show correlation analysis.")

        st.markdown("---")

        # Target Variable Analysis
        st.subheader("Target Variable Distribution")

        # Combine all variables for selection
        all_interact_variables = []
        for col in numerical_cols:
            all_interact_variables.append(f"{col} (Numerical)")
        for col in categorical_cols:
            all_interact_variables.append(f"{col} (Categorical)")

        if all_interact_variables:
            selected_interact_variable = st.selectbox(
                "Select a variable to visualise its distribution against target variable:",
                all_interact_variables,
                key="variable_interact_select",
            )

            # Extract variable name and type
            if selected_interact_variable:
                var_interact_name = selected_interact_variable.split(" (")[0]
                var_interact_type = (
                    "Numerical"
                    if "(Numerical)" in selected_interact_variable
                    else "Categorical"
                )
            else:
                var_interact_name = ""
                var_interact_type = ""

            if var_interact_type == "Numerical":
                # Histogram
                fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
                sns.histplot(
                    data=df,
                    x=var_interact_name,
                    ax=ax_hist,
                    color="skyblue",
                    hue=TARGET_COLUMN,
                    multiple="dodge",
                )
                ax_hist.set_title(
                    f"Distribution of {var_interact_name} by {TARGET_COLUMN}"
                )
                ax_hist.set_xlabel(var_interact_name)
                ax_hist.set_ylabel("Frequency")
                st.pyplot(fig_hist)

            else:  # Categorical
                # Cross-tabulation
                crosstab = pd.crosstab(
                    df[var_interact_name], df[TARGET_COLUMN], margins=True
                )
                st.write(f"Cross-tabulation: {var_interact_name} vs {TARGET_COLUMN}")
                st.dataframe(crosstab, use_container_width=True)

                # Stacked bar chart
                fig_stack, ax_stack = plt.subplots(figsize=(10, 6))
                crosstab_pct = (
                    pd.crosstab(
                        df[var_interact_name],
                        df[TARGET_COLUMN],
                        normalize="index",
                    )
                    * 100
                )
                crosstab_pct.plot(
                    kind="bar",
                    stacked=True,
                    ax=ax_stack,
                    color=["lightcoral", "lightblue"],
                )
                ax_stack.set_title(
                    f"{var_interact_name} Distribution by {TARGET_COLUMN} (%)"
                )
                ax_stack.set_xlabel(var_interact_name)
                ax_stack.set_ylabel("Percentage")
                ax_stack.legend(title=TARGET_COLUMN)
                ax_stack.tick_params(axis="x", rotation=0)
                plt.tight_layout()
                st.pyplot(fig_stack)
        else:
            st.info("No variables available for visualisation.")

    elif page == "Model Performance & Insights":
        st.header("📈 Model Performance & Feature Importance")
        st.markdown(
            "Noted that in this section, the model is trained on 80% of the data, and the remaining 20% is used for evaluation."
        )

        # Make predictions on the validation data for evaluation
        y_pred_proba = part_model.predict_proba(X_test_imputed)[:, 1]
        y_pred = part_model.predict(X_test_imputed)

        st.subheader("Classification Report")
        # Convert classification report to DataFrame for better display
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Create a DataFrame from the classification report
        report_df = pd.DataFrame(report_dict).transpose().round(3)

        # Calculate ROC AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", round(float(accuracy_score(y_test, y_pred)), 3))
        with col2:
            st.metric("Precision", round(float(precision_score(y_test, y_pred)), 3))
        with col3:
            st.metric("Recall", round(float(recall_score(y_test, y_pred)), 3))
        with col4:
            st.metric("F1 Score", round(float(f1_score(y_test, y_pred)), 3))
        with col5:
            st.metric("ROC AUC", round(float(roc_auc), 3))

        # Display as a table
        st.dataframe(report_df, use_container_width=True)

        st.subheader("ROC AUC Curve")

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
        ax_roc.set_ylim(0.0, 1.0)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

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
