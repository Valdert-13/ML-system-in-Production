import pytest
import os

@pytest.mark.files
def test_file1_method1():
    list_file_fact = os.listdir('./')
    list_file_must = ['config.py',
                      'config.yaml',
                      'dataset_builder.py',
                      'main.py',
                      'model_leaning.py',
                      'predict_data.py',
                      'preprocessing.py']

    list_file = list(set(list_file_fact) & set(list_file_must))

    assert len(list_file) == len(list_file_must),'test failed, check the required file list'

