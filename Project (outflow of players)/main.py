from config import *
from dataset_builder import build_dataset_raw
from preprocessing import *
import pandas as pd
import pickle
from model_leaning import *
from predict_data import get_predict

if REBUILD_TRAIN:
    """Creating the train dataset (config.yaml (REBUILD_TRAIN == True))"""
    build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                      churned_end_date=CHURNED_END_DATE,
                      inter_list=INTER_LIST,
                      raw_data_path=RAW_TRAIN_PATH,
                      dataset_path=DATASET_PATH,
                      mode='train')

if REBUILD_TEST:
    """Creating the test dataset (config.yaml (REBUILD_TEST == True))"""
    build_dataset_raw(churned_start_date = CHURNED_START_DATE,
                      churned_end_date  = CHURNED_END_DATE,
                      inter_list=INTER_LIST,
                      raw_data_path=RAW_TEST_PATH,
                      dataset_path=DATASET_PATH,
                      mode='test')

train_dataset = pd.read_csv(TRAIN_DATA_FILEPATH, sep=SEP)

train = prepare_dataset(dataset = train_dataset,
                        mode = 'train',
                        dataset_path = DATASET_PATH,
                        INTER_LIST = INTER_LIST)

X_train, X_test, y_train, y_test = scale_balance(train)

if LOAD_MODEL:
    try:
        with open(MODEL_NAME, 'rb') as input_file:
            model = pickle.load(input_file)
    except IOError as exp:
        with open("config.yaml", 'r') as stream:
            conf = yaml.safe_load(stream)
            conf['LOAD_MODEL'] = False
        with open("config.yaml", 'w') as stream:
            yaml.dump(conf, stream)
        print(exp)

else:
    model = xgb_fit(X_train, y_train)
    with open(MODEL_NAME, 'wb') as f:
        pickle.dump(model, f)

predict_proba_test = model.predict_proba(X_test)
predict_test = model.predict(X_test)
precision, recall, f1, ll, roc_auc = evaluation(y_test, predict_test, predict_proba_test[:, 1])

test_dataset = pd.read_csv(TEST_DATA_FILEPATH, sep=SEP)

test = prepare_dataset(dataset = test_dataset,
                        mode = 'train',
                        dataset_path = DATASET_PATH,
                        INTER_LIST = INTER_LIST)

test_balance = scale_test(test)

get_predict(model, test_balance)