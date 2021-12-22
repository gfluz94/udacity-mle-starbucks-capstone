#!/usr/bin/env python3

import os
import argparse
from typing import List
import logging
import pandas as pd
import numpy as np
import scipy.stats as st
import dill

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             f1_score, roc_auc_score, cohen_kappa_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s")
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ColumnSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols: list):
        """ Custom sklearn transformer to select a set of columns.

        Attributes:
            cols (list of str) representing the columns to be selected 
            in a pandas DataFrame.
                
        """
        self.__cols = cols
        self.__pd_df = pd.DataFrame
        
    @property
    def cols(self):
        return self.__cols
    
    def get_feature_names(self):
        return self.__cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        return X.loc[:, self.__cols]
    

class NumericImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, method: str = "mean", fill_value=None):
        """ Custom sklearn transformer to impute numeric data when it is missing.

        Attributes:
            method (str) representing the method (mean/median/constant)
            fill_value (int/float) representing the constant value to be imputed 
                
        """
        assert method in ["mean", "median", "constant"], \
               "Allowed methods are `mean`, `median`, `constant`"
        if method == "constant":
            assert fill_value is not None, "Fill value must be provided for `constant`"
        self.__method = method
        self.__fill_value = fill_value
        self.__learned_values = {}
        self.__cols = []
        self.__pd_df = pd.DataFrame
        self.__np_mean = np.mean
        self.__np_median = np.median
        
    @property
    def method(self):
        return self.__method
    
    @property
    def fill_value(self):
        return self.__fill_value
    
    @property
    def learned_values(self):
        return self.__learned_values
    
    def __define_func(self):
        if self.__method == "mean":
            return self.__np_mean
        elif self.__method == "median":
            return self.__np_median
        
    def get_feature_names(self):
        return self.__cols
    
    def fit(self, X, y=None):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        self.__cols = X_.columns
        if self.__method in ["mean", "median"]:
            func = self.__define_func()
            for column in X_.columns:
                self.__learned_values[column] = func(X_.loc[~X_[column].isnull(), column])
        elif self.__method == "constant":
            for column in X_.columns:
                self.__learned_values[column] = self.__fill_value
        return self
    
    def transform(self, X):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        for column in X_.columns:
            X_.loc[X_[column].isnull(), column] = self.__learned_values[column]
        return X_
    

class CategoricalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, method: str = "most_frequent", fill_value=None):
        """ Custom sklearn transformer to impute categorical data when it is missing.

        Attributes:
            method (str) representing the method (most_frequent/constant)
            fill_value (int/str) representing the constant value to be imputed 
                
        """
        assert method in ["most_frequent", "constant"], \
               "Allowed methods are `most_frequent`, `constant`"
        if method == "constant":
            assert fill_value is not None, "Fill value must be provided for `constant`"
        self.__method = method
        self.__fill_value = fill_value
        self.__learned_values = {}
        self.__cols = []
        self.__pd_df = pd.DataFrame
        
    @property
    def method(self):
        return self.__method
    
    @property
    def fill_value(self):
        return self.__fill_value
    
    @property
    def learned_values(self):
        return self.__learned_values
    
    def get_feature_names(self):
        return self.__cols
    
    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        self.__cols = X_.columns
        if self.__method == "most_frequent":
            for column in X_.columns:
                self.__learned_values[column] = X_.loc[:, column].value_counts(ascending=False).index[0]
        elif self.__method == "constant":
            for column in X_.columns:
                self.__learned_values[column] = self.__fill_value
        return self
    
    def transform(self, X):
        assert isinstance(X, self.__pd_df), "`X` should be a pandas dataframe"
        X_ = X.copy()
        for column in X_.columns:
            X_.loc[X_[column].isnull(), column] = self.__learned_values[column]
        return X_


def create_preprocessor(numerical_features: List[str], categorical_features: List[str], 
                        categorical_encoded_features: List[str], numerical_imputation_strategy: str,
                        categorical_imputation_strategy: str) -> Pipeline:
    """
    Function that creates object responsible for preprocessing the dataset

        Args: 
            numerical_features (list of str): numerical features' column names
            categorical_features (list of str): categorical features' column names - one hot encoding will be performed
            categorical_encoded_features (list of str): categorical features' column names - one hot encoding won't be performed
            numerical_imputation_strategy (str): imputation strategy for numerical columns (mean|median)
            categorical_imputation_strategy (str): imputation strategy for categorical columns (most_frequent|any str)

        Returns: 
           sklearn.pipeline.Pipeline object with `fit` and `transform` methods
    """
    num_preprocessing = Pipeline(steps=[
        ("num_cols", ColumnSelector(cols=numerical_features)),
        ("num_imputer", NumericImputer(method=numerical_imputation_strategy)),
        ("std_scaler", StandardScaler())
    ])

    strategy = categorical_imputation_strategy
    if categorical_imputation_strategy != "most_frequent":
        strategy = "constant"

    cat_preprocessing = Pipeline(steps=[
        ("cat_cols", ColumnSelector(cols=categorical_features)),
        ("cat_imputer", CategoricalImputer(method=strategy, fill_value=categorical_imputation_strategy)),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    cat_encoded_preprocessing = Pipeline(steps=[
        ("cat_encoded_cols", ColumnSelector(cols=categorical_encoded_features)),
        ("cat_encoded_imputer", CategoricalImputer(method=strategy, fill_value=categorical_imputation_strategy)),
    ])

    preprocessor = FeatureUnion(transformer_list=[
        ("num_preprocessing", num_preprocessing),
        ("cat_preprocessing", cat_preprocessing),
        ("cat_encoded_preprocessing", cat_encoded_preprocessing),
    ])

    return preprocessor


def get_data(path: str, validation: bool = False):
    """
    Function that retrieves data according to path

        Args: 
            path (str): full path to files
            validation (bool): if True, validation set will be returned as long with train set

        Returns: 
           pandas.DataFrame or tuple of pandas.DataFrame containing data
    """
    if validation:
        df_train = pd.read_csv(os.path.join(path, "train"))
        df_val = pd.read_csv(os.path.join(path, "validation"))
        return df_train, df_val
    df = pd.concat([pd.read_csv(os.path.join(path, file)) for file in os.listdir(path)], axis=0, ignore_index=True)
    return df


def metric_function(metric_name: str):
    """
    Function that retrieves method according to classification metric name

        Args: 
            metric_name (str): valid classification metric name
        Returns: 
           Method that accepts `y_true` and `y_pred`/`y_pred` to compute metric value
    """
    if metric_name == "ACCURACY":
        return accuracy_score
    elif metric_name == "RECALL":
        return recall_score
    elif metric_name == "PRECISION":
        return precision_score
    elif metric_name == "F1-SCORE":
        return f1_score
    elif metric_name == "ROC-AUC":
        return roc_auc_score
    elif metric_name == "COHEN-KAPPA":
        return cohen_kappa_score
    else:
        raise ValueError(f"No metric available under the name of {metric_name}.")

def evaluate_model(inference_pipeline: Pipeline, df: pd.DataFrame, target_name: str):
    """
    Function that evaluates inferece pipeline performance on a held-out dataset.
    Bootstrapping (20 trials) is applied to generate confidence intervals for metrics.

        Args: 
            inference_pipeline (sklearn.pipeline.Pipeline): pipeline to make inference on data
            df (pandas.DataFrame): dataframe containing data
            target_name (str): target variable column name

        Returns: 
           Dictionary with keys corresponding to classification metric names and values to their associated confidence intervals
    """
    n_experiments = 20
    confidence = 0.95
    
    metrics_dict = {
        "ACCURACY": [],
        "F1-SCORE": [],
        "PRECISION": [],
        "RECALL": [],
        "ROC-AUC": [],
        "COHEN-KAPPA": []
    }

    for _ in range(n_experiments):
        df_bootstrap = df.sample(frac=1.0, replace=True)
        X_val, y_val = df_bootstrap.drop(columns=[target_name]), df_bootstrap.loc[:, target_name].values
        y_pred = inference_pipeline.predict(X_val)
        y_proba = inference_pipeline.predict_proba(X_val)[:, 1]

        for key in metrics_dict.keys():
            if key == "ROC-AUC":
                metrics_dict[key].append(metric_function(key)(y_val, y_proba))
            else:
                metrics_dict[key].append(metric_function(key)(y_val, y_pred))

    output_results = {}
    for key, value in metrics_dict.items(): 
        lower, upper = st.t.interval(confidence, n_experiments-1, loc=np.mean(value), scale=st.sem(value))
        avg_value = np.round((lower + upper) / 2, 4)
        delta = np.round(upper - avg_value, 4)
        output_results[key] = f"{avg_value} ± {delta}"

    return output_results


if __name__ == "__main__":

    logger.info("Training Job started...")
    
    # Reading arguments from the command line
    logger.info("Setting hyperparameters...")
    parser = argparse.ArgumentParser(description="Input hyperparameters for model training.")
    parser.add_argument("--target-name", metavar="N", type=str, 
                        help="Target variable column name.",
                        default="responded_customer")
    parser.add_argument("--numerical-features", metavar="N", type=str, 
                        nargs='+', help="Numerical features to be preprocessed.",
                        default=[
                            "age", "membership_duration", "reward", "difficulty", 
                            "duration", "average_purchase", "frequency"
                        ])
    parser.add_argument("--categorical-features-oh", metavar="N", type=str, 
                        nargs='+', help="Categorical features to be preprocessed with one-hot encoding.",
                        default=["gender", "offer_type"])
    parser.add_argument("--categorical-features", metavar="N", type=str, 
                        nargs='+', help="Categorical features to be preprocessed without one-hot encoding.",
                        default=["web", "email", "mobile", "social"])
    parser.add_argument("--oversampling", metavar="N", type=bool,
                        help="True if SMOTE oversampling technique must be carried out.",
                        default=True)
    parser.add_argument("--resampling-ratio", metavar="N", type=float,
                        help="Resampling ratio to perform SMOTE.",
                        default=0.7)
    parser.add_argument("--numerical-imputation", metavar="N", type=str,
                        help="Method to impute numerical data: mean/median.",
                        default="median")
    parser.add_argument("--categorical-imputation", metavar="N", type=str,
                        help="Method to impute categorical data: most_frequent/value.",
                        default="most_frequent")
    parser.add_argument("--seed", metavar="N", type=int,
                        help="Random seed to ensure reproducibility.",
                        default=99)
    parser.add_argument("--xgboost-objective", metavar="N", type=str,
                        help="XGBoost objective function.",
                        default="binary:logistic")
    parser.add_argument("--xgboost-eval-metric", metavar="N", type=str,
                        help="XGBoost evaluation metric.",
                        default="error")
    parser.add_argument("--xgboost-estimators", metavar="N", type=int,
                        help="XGBoost number of estimators.",
                        default=100)
    parser.add_argument("--xgboost-max-depth", metavar="N", type=int,
                        help="XGBoost trees' maximum depth.",
                        default=3)
    parser.add_argument("--xgboost-gamma", metavar="N", type=float,
                        help="XGBoost gamma parameter.",
                        default=0)
    parser.add_argument("--xgboost-learning-rate", metavar="N", type=float,
                        help="XGBoost gamma parameter.",
                        default=0.1)
    parser.add_argument('--model-dir', metavar="N", type=str,
                        help="Full path to save model artifacts.",
                        default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument('--train-data', metavar="N", type=str,
                        help="Full path to retrieve training data.",
                        default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--test-data', metavar="N", type=str,
                        help="Full path to retrieve test data.",
                        default=os.environ["SM_CHANNEL_TEST"])
    args = parser.parse_args()
    logger.info("Hyperparameters defined!")

    # Log hyperparameters to be tuned
    logger.info(f"Resampling ratio: {args.resampling_ratio}")
    logger.info(f"Numerical imputation method: {args.numerical_imputation}")
    logger.info(f"Categorical imputation method: {args.categorical_imputation}")
    logger.info(f"XGBoost number of estimators: {args.xgboost_estimators}")
    logger.info(f"XGBoost trees' maximum depth: {args.xgboost_max_depth}")
    logger.info(f"XGBoost gamma: {args.xgboost_gamma}")
    logger.info(f"XGBoost learning rate: {args.xgboost_learning_rate}")

    # Get train and validation set
    logger.info("Importing data...")
    df_train = get_data(args.train_data)
    X_train, y_train = df_train.drop(columns=[args.target_name]), df_train.loc[:, args.target_name].values
    df_test = get_data(args.test_data)
    logger.info("Data defined!")

    # Define preprocessing step
    preprocessor = create_preprocessor(
        numerical_features=args.numerical_features,
        categorical_features=args.categorical_features_oh,
        categorical_encoded_features=args.categorical_features,
        numerical_imputation_strategy=args.numerical_imputation,
        categorical_imputation_strategy=args.categorical_imputation,
    )

    # Preprocess data
    logger.info("Preprocessing data...")
    X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
    logger.info("Data preprocessed!")

    # Define oversampling step, if carried out
    scale_pos_weight = (1-y_train).sum() / y_train.sum()
    if args.oversampling:
        logger.info("Performing oversampling...")
        smote = SMOTE(sampling_strategy=args.resampling_ratio)
        X_train_preprocessed, y_train = smote.fit_resample(X_train_preprocessed, y_train)
        scale_pos_weight = 1/args.resampling_ratio
        logger.info("Oversampling completed!")

    # Define XGBoost model
    xgb = XGBClassifier(
        objective=args.xgboost_objective,
        eval_metric=args.xgboost_eval_metric,
        n_estimators=args.xgboost_estimators,
        max_depth=args.xgboost_max_depth,
        gamma=args.xgboost_gamma,
        learning_rate=args.xgboost_learning_rate,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=args.seed
    )

    # Fit model
    logger.info("Training algorithm...")
    xgb.fit(X_train_preprocessed, y_train)
    logger.info("Algorithm successfully trained!")

    # Build complete inference pipeline
    inference_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("xgboost", xgb)
    ])

    # Predict and evaluate on test set
    logger.info("Evaluating performance...")
    metrics_results = evaluate_model(inference_pipeline, df_test, args.target_name)
    for metric_name, metric_value in metrics_results.items():
        logger.info(f"{metric_name} = {metric_value}")
    logger.info("Performance evaluated!")

    # Save inference pipeline
    logger.info("Saving model artifacts...")
    with open(os.path.join(args.model_dir, "model.pkl"), "wb") as file:
        dill.dump(inference_pipeline, file)
    logger.info("Model artifacts successfully saved!")

    logger.info("Training job successfully ended!")