"""
Script with functionality to save and load evaluation results.
"""

import pandas as pd
import json

def save_evaluation_results_to_json(evaluation_results: dict, file_path: str):
    """
    Saves evaluation results to a JSON file.

    Args:
        evaluation_results (dict): Dictionary containing evaluation results (each a pandas DataFrame).
        file_path (str): Path to the JSON file where results will be saved.
    """
    save_dict = {k: v.to_dict(orient='records') for k, v in evaluation_results.items()}
    with open(file_path, 'w') as f:
        json.dump(save_dict, f, indent=4)

def load_evaluation_results_from_json(file_path: str) -> dict:
    """
    Loads evaluation results from a JSON file.

    Args:
        file_path (str): Path to the JSON file from which results will be loaded.

    Returns:
        dict: Dictionary containing evaluation results (each a pandas DataFrame).
    """
    with open(file_path, 'r') as f:
        loaded_dict = json.load(f)
    
    evaluation_results = {k: pd.DataFrame(v) for k, v in loaded_dict.items()}
    return evaluation_results