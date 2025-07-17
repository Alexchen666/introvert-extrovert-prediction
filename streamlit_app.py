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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Set page config for better layout
st.set_page_config(layout="wide", page_title="Data Science Project Demo")

# Global Variables and Constants
TARGET_COLUMN = "Personality"


# Data Loading
@st.cache_data  # Cache data loading to avoid re-running on every interaction
def load_data(train_path, test_path):
    """Loads training and testing datasets."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except FileNotFoundError:
        st.error(
            f"Error: Make sure '{train_path}' and '{test_path}' are in the same directory."
        )
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()


# Data Preprocessing
def create_preprocessor(train_df):
    """
    Creates and fits a preprocessing pipeline based on the training data.
    Handles numerical and categorical features, including imputation and encoding.
    """
    # Identify numerical and categorical features
    numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_df.select_dtypes(include="object").columns.tolist()

    # Remove target column from features if present
    if TARGET_COLUMN in numerical_cols:
        numerical_cols.remove(TARGET_COLUMN)
    if TARGET_COLUMN in categorical_cols:  # Should not happen if target is numerical
        categorical_cols.remove(TARGET_COLUMN)

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="passthrough",
    )
    return preprocessor, numerical_cols, categorical_cols


def preprocess_data(df, preprocessor, fit_preprocessor=False):
    """Applies the preprocessing pipeline to the dataframe."""
    if fit_preprocessor:
        # Fit and transform (for training data)
        X_processed = preprocessor.fit_transform(df)
    else:
        # Transform only (for test or new prediction data)
        X_processed = preprocessor.transform(df)
    return X_processed


# Model Training
def train_model(X_train_processed, y_train, **kwargs):
    """Trains an XGBoost Classifier model with optional hyperparameters."""
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": accuracy_score,
        "n_estimators": 100,
        "random_state": 42,
    }
    # Update default params with any provided kwargs
    default_params.update(kwargs)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train_processed, y_train)
    return model


# Hyperparameter Tuning Functions
def perform_hyperparameter_tuning(
    X_train_processed, y_train, param_grid, search_type="grid", cv_folds=5, n_iter=10
):
    """
    Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Args:
        X_train_processed: Preprocessed training features
        y_train: Training target
        param_grid: Dictionary of hyperparameters to tune
        search_type: "grid" for GridSearchCV, "random" for RandomizedSearchCV
        cv_folds: Number of cross-validation folds
        n_iter: Number of iterations for RandomizedSearchCV

    Returns:
        best_model: The best model found
        search_results: The search object with all results
    """
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="accuracy",
        random_state=42,
        enable_categorical=True,  # Enable categorical support in XGBoost
    )

    if search_type == "grid":
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
    else:  # random search
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

    search.fit(X_train_processed, y_train)
    return search.best_estimator_, search


def get_hyperparameter_grid():
    """Returns a comprehensive hyperparameter grid for XGBoost."""
    return {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0, 0.1, 0.5, 1.0],
    }


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
            "Hyperparameter Tuning",
            "Batch Prediction",
        ],
    )

    train_df, test_df = load_data("data/train.csv", "data/test.csv")

    # Ensure the target column exists in the training data
    if TARGET_COLUMN not in train_df.columns:
        st.error(
            f"Error: The target column '{TARGET_COLUMN}' was not found in 'train.csv'. Please adjust the TARGET_COLUMN variable in the code."
        )
        st.stop()

    # Separate features and target for training
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    # Encode string labels to numeric for XGBoost binary classification
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Store label encoder in session state for batch predictions
    st.session_state.label_encoder = label_encoder

    # Create and fit the preprocessor
    preprocessor, numerical_cols, categorical_cols = create_preprocessor(X_train)
    X_train_processed = preprocess_data(X_train, preprocessor, fit_preprocessor=True)

    # Get feature names after one-hot encoding
    # This part is a bit tricky with ColumnTransformer. We need to get the names
    # after the one-hot encoder has been fitted.
    try:
        ohe_feature_names = (
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_cols)
        )
        processed_feature_names = numerical_cols + list(ohe_feature_names)
    except Exception:
        # Fallback if get_feature_names_out fails (e.g., no categorical columns)
        processed_feature_names = (
            numerical_cols + categorical_cols
        )  # Simple fallback, might not be accurate for OHE

    # Train the model
    model = train_model(X_train_processed, y_train_encoded)

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
        st.markdown("Here's a glimpse of the raw training and testing datasets.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Data (`train.csv`)")
            st.write(train_df.head())
            st.write(f"Shape: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
            st.write("---")
            st.subheader("Training Data Info")
            buffer = io.StringIO()
            train_df.info(buf=buffer)
            st.text(buffer.getvalue())

        with col2:
            st.subheader("Test Data (`test.csv`)")
            st.write(test_df.head())
            st.write(f"Shape: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
            st.write("---")
            st.subheader("Test Data Info")
            buffer = io.StringIO()
            test_df.info(buf=buffer)
            st.text(buffer.getvalue())

        st.markdown("---")
        st.subheader("Missing Values Overview")
        st.write("Percentage of missing values in training data:")
        missing_train = train_df.isnull().sum() / len(train_df) * 100
        st.dataframe(
            missing_train[missing_train > 0]
            .sort_values(ascending=False)
            .to_frame(name="Missing %")
        )

        st.write("Percentage of missing values in test data:")
        missing_test = test_df.isnull().sum() / len(test_df) * 100
        st.dataframe(
            missing_test[missing_test > 0]
            .sort_values(ascending=False)
            .to_frame(name="Missing %")
        )

    elif page == "Model Performance & Insights":
        st.header("📈 Model Performance & Feature Importance")
        st.markdown(
            "This section provides insights into the trained model's performance and explains its predictions using SHAP values."
        )

        # Make predictions on the training data for evaluation (or a held-out validation set if available)
        # For simplicity in this demo, we'll evaluate on training data, but in a real scenario,
        # you'd use a separate validation/test set.
        y_pred_proba = model.predict_proba(X_train_processed)[:, 1]
        y_pred = model.predict(X_train_processed)

        st.subheader("ROC AUC Curve")
        st.markdown(
            "The Receiver Operating Characteristic (ROC) curve illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied."
        )

        fpr, tpr, thresholds = roc_curve(y_train_encoded, y_pred_proba)
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
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

        st.subheader("Classification Report")
        st.text(classification_report(y_train_encoded, y_pred))

        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            confusion_matrix(y_train_encoded, y_pred),
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
        shap_values = explainer.shap_values(X_train_processed)

        # Ensure that processed_feature_names matches the shap_values dimensions
        if len(processed_feature_names) != X_train_processed.shape[1]:
            st.warning(
                "Mismatch between feature names and processed data columns. SHAP plot might be inaccurate."
            )
            # Fallback for feature names if `get_feature_names_out` failed or columns were dropped
            processed_feature_names = [
                f"feature_{i}" for i in range(X_train_processed.shape[1])
            ]

        # SHAP Summary Plot
        fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_train_processed,
            feature_names=processed_feature_names,
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

    elif page == "Hyperparameter Tuning":
        st.header("🔧 Hyperparameter Tuning")
        st.markdown("""
            This section allows you to experiment with different hyperparameters for the XGBoost model.
            Enter comma-separated values for each parameter you want to tune.
        """)

        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Hyperparameter Selection")

            # Search type selection
            search_type = st.selectbox(
                "Search Type",
                ["grid", "random"],
                help="Grid Search: Exhaustive search over all parameter combinations. Random Search: Random sampling of parameter combinations.",
            )

            # Cross-validation folds
            cv_folds = st.slider(
                "Cross-Validation Folds", min_value=3, max_value=10, value=5
            )

            if search_type == "random":
                n_iter = st.slider(
                    "Number of Iterations", min_value=10, max_value=100, value=20
                )

            st.subheader("Parameter Ranges")
            st.markdown(
                "*Enter comma-separated values for each parameter you want to tune. Leave empty to use default value.*"
            )

            # Helper function to parse comma-separated values
            def parse_values(input_str, value_type=float):
                if not input_str.strip():
                    return []
                try:
                    if isinstance(value_type, int):
                        return [int(x.strip()) for x in input_str.split(",")]
                    else:
                        return [float(x.strip()) for x in input_str.split(",")]
                except ValueError:
                    st.error(
                        f"Invalid input format. Please enter comma-separated {value_type.__name__} values."
                    )
                    return []

            # n_estimators
            n_estimators_input = st.text_input(
                "Number of Estimators",
                placeholder="e.g., 100, 200, 300",
                help="Number of boosting rounds. Higher values may improve performance but increase training time.",
            )

            # max_depth
            max_depth_input = st.text_input(
                "Max Depth",
                placeholder="e.g., 3, 4, 5, 6",
                help="Maximum depth of trees. Controls model complexity.",
            )

            # learning_rate
            learning_rate_input = st.text_input(
                "Learning Rate",
                placeholder="e.g., 0.01, 0.1, 0.2",
                help="Step size shrinkage to prevent overfitting.",
            )

            # subsample
            subsample_input = st.text_input(
                "Subsample",
                placeholder="e.g., 0.8, 0.9, 1.0",
                help="Fraction of samples used for training each tree.",
            )

            # colsample_bytree
            colsample_input = st.text_input(
                "Column Sample by Tree",
                placeholder="e.g., 0.8, 0.9, 1.0",
                help="Fraction of features used for training each tree.",
            )

            # Regularization parameters
            reg_alpha_input = st.text_input(
                "L1 Regularization (reg_alpha)",
                placeholder="e.g., 0, 0.1, 0.5",
                help="L1 regularization term on weights.",
            )

            reg_lambda_input = st.text_input(
                "L2 Regularization (reg_lambda)",
                placeholder="e.g., 0, 0.1, 0.5",
                help="L2 regularization term on weights.",
            )

        with col2:
            st.subheader("Tuning Results")

            # Build parameter grid
            param_grid = {}

            # Parse input values
            if n_estimators_input:
                n_estimators_values = parse_values(n_estimators_input, int)
                if n_estimators_values:
                    param_grid["n_estimators"] = n_estimators_values

            if max_depth_input:
                max_depth_values = parse_values(max_depth_input, int)
                if max_depth_values:
                    param_grid["max_depth"] = max_depth_values

            if learning_rate_input:
                learning_rate_values = parse_values(learning_rate_input, float)
                if learning_rate_values:
                    param_grid["learning_rate"] = learning_rate_values

            if subsample_input:
                subsample_values = parse_values(subsample_input, float)
                if subsample_values:
                    param_grid["subsample"] = subsample_values

            if colsample_input:
                colsample_values = parse_values(colsample_input, float)
                if colsample_values:
                    param_grid["colsample_bytree"] = colsample_values

            if reg_alpha_input:
                reg_alpha_values = parse_values(reg_alpha_input, float)
                if reg_alpha_values:
                    param_grid["reg_alpha"] = reg_alpha_values

            if reg_lambda_input:
                reg_lambda_values = parse_values(reg_lambda_input, float)
                if reg_lambda_values:
                    param_grid["reg_lambda"] = reg_lambda_values

            # Show current parameter grid
            if param_grid:
                st.subheader("Current Parameter Grid:")
                st.json(param_grid)

                # Calculate total combinations for grid search
                if search_type == "grid":
                    total_combinations = 1
                    for values in param_grid.values():
                        total_combinations *= len(values)
                    st.info(f"Total combinations to test: {total_combinations}")

                    if total_combinations > 100:
                        st.warning(
                            "⚠️ Large number of combinations detected. Consider using Random Search or reducing parameter ranges."
                        )

            # Start tuning button
            if st.button("🚀 Start Hyperparameter Tuning", type="primary"):
                if not param_grid:
                    st.error("Please enter at least one hyperparameter range!")
                else:
                    with st.spinner(
                        "Performing hyperparameter tuning... This may take a while."
                    ):
                        try:
                            # Perform hyperparameter tuning
                            kwargs = (
                                {"n_iter": n_iter} if search_type == "random" else {}
                            )
                            best_model, search_results = perform_hyperparameter_tuning(
                                X_train_processed,
                                y_train_encoded,
                                param_grid,
                                search_type=search_type,
                                cv_folds=cv_folds,
                                **kwargs,
                            )

                            # Store results in session state
                            st.session_state.tuning_results = {
                                "best_model": best_model,
                                "search_results": search_results,
                                "best_params": search_results.best_params_,
                                "best_score": search_results.best_score_,
                            }

                            st.success("Hyperparameter tuning completed!")

                        except Exception as e:
                            st.error(f"Error during hyperparameter tuning: {str(e)}")

            # Display results if available
            if hasattr(st.session_state, "tuning_results"):
                results = st.session_state.tuning_results

                st.subheader("🏆 Best Parameters Found:")
                st.json(results["best_params"])

                st.subheader("📊 Best Cross-Validation Score:")
                st.metric("Accuracy", f"{results['best_score']:.4f}")

                # Compare with baseline model
                st.subheader("📈 Model Comparison")

                # Train baseline model
                baseline_model = train_model(X_train_processed, y_train_encoded)
                baseline_scores = cross_val_score(
                    baseline_model,
                    X_train_processed,
                    y_train_encoded,
                    cv=cv_folds,
                    scoring="accuracy",
                )

                # Get tuned model scores
                tuned_scores = cross_val_score(
                    results["best_model"],
                    X_train_processed,
                    y_train_encoded,
                    cv=cv_folds,
                    scoring="accuracy",
                )

                comparison_df = pd.DataFrame(
                    {
                        "Model": ["Baseline", "Tuned"],
                        "Mean CV Score": [
                            f"{baseline_scores.mean():.4f}",
                            f"{tuned_scores.mean():.4f}",
                        ],
                        "Std CV Score": [
                            f"{baseline_scores.std():.4f}",
                            f"{tuned_scores.std():.4f}",
                        ],
                        "Improvement": [
                            "--",
                            f"{((tuned_scores.mean() - baseline_scores.mean()) / baseline_scores.mean() * 100):+.2f}%",
                        ],
                    }
                )

                st.dataframe(comparison_df, use_container_width=True)

                # Show top parameter combinations
                st.subheader("🔍 Top Parameter Combinations")

                # Create a DataFrame from CV results
                cv_results_df = pd.DataFrame(results["search_results"].cv_results_)

                # Show top 5 parameter combinations
                top_results = cv_results_df.nlargest(5, "mean_test_score")[
                    ["params", "mean_test_score", "std_test_score"]
                ]

                for idx, (_, row) in enumerate(top_results.iterrows(), 1):
                    with st.expander(
                        f"Rank #{idx}: Score {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})"
                    ):
                        st.json(row["params"])

                # Feature importance comparison
                st.subheader("🎯 Feature Importance Comparison")

                # Get feature importances
                baseline_importance = baseline_model.feature_importances_
                tuned_importance = results["best_model"].feature_importances_

                # Create comparison plot
                fig_importance, ax = plt.subplots(figsize=(12, 6))

                x_pos = np.arange(len(baseline_importance))
                width = 0.35

                ax.bar(
                    x_pos - width / 2,
                    baseline_importance,
                    width,
                    label="Baseline Model",
                    alpha=0.7,
                    color="skyblue",
                )
                ax.bar(
                    x_pos + width / 2,
                    tuned_importance,
                    width,
                    label="Tuned Model",
                    alpha=0.7,
                    color="lightcoral",
                )

                ax.set_xlabel("Features")
                ax.set_ylabel("Importance")
                ax.set_title("Feature Importance Comparison: Baseline vs Tuned Model")
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"F{i}" for i in range(len(baseline_importance))])
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig_importance)

                # Option to use tuned model for predictions
                if st.button("✅ Use Tuned Model for Future Predictions"):
                    st.session_state.use_tuned_model = True
                    st.session_state.current_model = results["best_model"]
                    st.success("🎉 Tuned model is now set as the active model!")

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
                train_feature_cols = X_train.columns.tolist()

                # Add any missing columns to new_data_df and fill with NaN
                for col in train_feature_cols:
                    if col not in new_data_df.columns:
                        new_data_df[col] = np.nan

                # Drop any extra columns in new_data_df that were not in training features
                new_data_df = new_data_df[train_feature_cols]

                X_new_processed = preprocess_data(
                    new_data_df, preprocessor, fit_preprocessor=False
                )

                # Make predictions
                predictions_encoded = model.predict(X_new_processed)
                prediction_proba = model.predict_proba(X_new_processed)[:, 1]

                # Decode predictions back to original string labels
                if hasattr(st.session_state, "label_encoder"):
                    predictions_decoded = (
                        st.session_state.label_encoder.inverse_transform(
                            predictions_encoded
                        )
                    )
                else:
                    # Fallback if label encoder not available
                    predictions_decoded = predictions_encoded

                # Add predictions to the original new_data_df
                new_data_df["Predicted_" + TARGET_COLUMN] = predictions_decoded
                new_data_df["Prediction_Probability"] = prediction_proba

                st.subheader("Prediction Results (Head)")
                st.write(new_data_df.head())

                # Provide download link
                csv_output = new_data_df.to_csv(index=False).encode("utf-8")
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
