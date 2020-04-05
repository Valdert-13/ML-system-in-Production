import yaml

with open("config.yaml", 'r') as stream:
    try:
        conf = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

TRAIN_DATA_FILEPATH = conf['TRAIN_DATA_FILEPATH']
TEST_DATA_FILEPATH = conf['TEST_DATA_FILEPATH']
SEP = conf['SEP']
RAW_TRAIN_PATH = conf['RAW_TRAIN_PATH']
RAW_TEST_PATH = conf['RAW_TEST_PATH']
DATASET_PATH = conf['DATASET_PATH']
OUTPUT_DATA_PATH = conf['OUTPUT_DATA_PATH']
REBUILD_TRAIN = conf['REBUILD_TRAIN']
REBUILD_TEST = conf['REBUILD_TEST']
CHURNED_START_DATE = conf['CHURNED_START_DATE']
CHURNED_END_DATE = conf['CHURNED_END_DATE']
INTER_LIST = [conf['INTER_1'], conf['INTER_2'], conf['INTER_3'], conf['INTER_4']]
MODEL_NAME = conf['MODEL_NAME']
LOAD_MODEL = conf['LOAD_MODEL']

