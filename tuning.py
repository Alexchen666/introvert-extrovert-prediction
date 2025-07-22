from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/train.csv")

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
def map_data(df: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
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


# Split the data into features and target
X = filter_data(df, TARGET_COLUMN, ID_COLUMN)[0]
y = df[TARGET_COLUMN].map(MAPPING_PERSONALITY)

# Identify numerical and categorical features
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include="object").columns.tolist()

X_processed = map_data(
    X,
    {cat_col: MAPPING_YES_NO for cat_col in categorical_cols},
)

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


# Hyperparameter Tuning Functions
def perform_hyperparameter_tuning(
    X_processed: pd.DataFrame, y_train: pd.Series, param_grid: dict, cv_folds: int = 5
) -> tuple:
    """
    Performs hyperparameter tuning using GridSearchCV.

    Args:
        X_processed (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Training target values.
        param_grid (dict): Dictionary of hyperparameters to tune.
        cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - best_model: The best model found during grid search
            - search_results: The GridSearchCV object with all results
    """
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=accuracy_score,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("classifier", base_model),
        ]
    )

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_processed, y_train)
    return search.best_estimator_, search


def get_hyperparameter_grid() -> dict:
    """
    Returns a comprehensive hyperparameter grid for XGBoost.

    Returns:
        dict: Dictionary containing hyperparameter names as keys and lists of values to try as values.
            Includes parameters for n_estimators, max_depth, learning_rate, subsample,
            colsample_bytree, reg_alpha, and reg_lambda.
    """
    return {
        "classifier__n_estimators": [50, 100, 200, 300],
        "classifier__max_depth": [3, 4, 5, 6, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2, 0.3],
        "classifier__subsample": [0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.8, 0.9, 1.0],
        "classifier__reg_alpha": [0, 0.1, 0.5, 1.0],
        "classifier__reg_lambda": [0, 0.1, 0.5, 1.0],
    }


if __name__ == "__main__":
    best_est, xgb_search = perform_hyperparameter_tuning(
        X_processed, y, get_hyperparameter_grid(), cv_folds=5
    )
    print("Best Parameters:", xgb_search.best_params_)
    print("Best Score:", xgb_search.best_score_)
