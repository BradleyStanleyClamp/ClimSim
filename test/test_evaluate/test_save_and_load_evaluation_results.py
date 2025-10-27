import evaluate
import pandas as pd
from pathlib import Path

def test_save_evaluation_results():
    dict_var = {
    'mlp': pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
    'cnn': pd.DataFrame({'x': [10, 20], 'y': [30, 40]})
    }

    evaluate.save_evaluation_results_to_json(dict_var, 'test_evaluation_results.json')

    path = Path('test_evaluation_results.json')
    assert path.exists()
    path.unlink()
    assert not path.exists()

def test_load_evaluation_results(results_path: str = 'test/unit_test_sets/evaluated_model_log/evaluation_comp_general/model_evaluation_results.json'):
    loaded_results = evaluate.load_evaluation_results_from_json(results_path)

    assert 'yus_mlp' in loaded_results
    assert isinstance(loaded_results['yus_mlp'], pd.DataFrame)
    assert not loaded_results['yus_mlp'].empty
