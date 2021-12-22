#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import dill
from io import StringIO

CSV_FORMAT = "text/csv"
JSON_FORMAT = "application/json"
MODEL_NAME = "model.pkl"


def input_fn(input_data, content_type):
    """
    Function that reads request input data accroding to supported data types

        Args: 
            input_data (str): raw input data stream
            content_type (str): content type (application/json or text/csv currently supported)

        Returns: 
           pandas.DataFrame that will be fed to the model for inference
    """
    if content_type == CSV_FORMAT:
        df = pd.read_csv(StringIO(input_data))
        return df
    elif content_type == JSON_FORMAT:
        input_dict = json.loads(input_data)
        df = pd.DataFrame([input_dict])
        return df
    else:
        raise ValueError(f"{content_type} currently not supported for inference!")


def model_fn(model_dir):
    """
    Function that loads previously trained model

        Args: 
            model_dir (str): path to stored model artifacts

        Returns: 
           Trained model object
    """
    with open(os.path.join(model_dir, MODEL_NAME), "rb") as file:
        inference_pipeline = dill.load(file)
    return inference_pipeline


def predict_fn(input_data, model):
    """
    Function that takes preprocessed request data and applies the loaded model

        Args: 
            input_data (pandas.DataFrame): dataframe returned by `input_fn`
            model (obj): model object returned by `predict_fn`

        Returns: 
           Probability array
    """
    probabilities = model.predict_proba(input_data)[:, 1]
    return probabilities


def output_fn(predictions, content_type):
    """
    Function that process output before it is sent to the end user

        Args: 
            predictions (numpy.array): probability array returned by `predict_fn`
            content_type (str): content type (application/json currently supported)

        Returns: 
           JSON formatted string containing predictions
    """
    assert content_type == JSON_FORMAT, f"Only content type {JSON_FORMAT} is supported!"
    response = {
        "response_probability": predictions.tolist()
    }
    return json.dumps(response)