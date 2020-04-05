from config import *
import pytest
import os

@pytest.mark.config
def test_path_data():
    list_path = [RAW_TRAIN_PATH, RAW_TEST_PATH, DATASET_PATH]
    for path in list_path:
        assert os.path.exists(path) == True,'the test failed, the path does not exist'

@pytest.mark.config
def test_data_sets():
    list_file_train = os.listdir(RAW_TRAIN_PATH)
    list_file_test = os.listdir(RAW_TEST_PATH)
    list_file_must = ['abusers.csv',
                      'logins.csv',
                      'nuclear.csv',
                      'payments.csv',
                      'pings.csv',
                      'profiles.csv',
                      'reports.csv',
                      'sample.csv',
                      'sessions.csv',
                      'shop.csv']

    list_train = list(set(list_file_train) & set(list_file_must))
    list_test = list(set(list_file_test) & set(list_file_must))
    assert len(list_train) == len(list_file_must),'test failed, check the required file list'
    assert len(list_test) == len(list_file_must),'test failed, check the required file list'

@pytest.mark.config
def test_config_rebuild_train():
    if not REBUILD_TRAIN:
        assert os.path.exists(TRAIN_DATA_FILEPATH) == True,\
            'the test failed, change the value of REBUILD_TRAIN in config.yaml to True'

@pytest.mark.config
def test_config_rebuild_test():
    if not REBUILD_TEST:
        assert os.path.exists(TEST_DATA_FILEPATH) == True, \
            'the test failed, change the value of TEST_DATA_FILEPATH in config.yaml to True'

@pytest.mark.config
def test_config_model():
    if LOAD_MODEL:
        assert os.path.exists(MODEL_NAME) == True, \
            'the test failed, change the value of LOAD_MODEL in config.yaml to False'