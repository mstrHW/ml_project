import os
import joblib
import json


def path_join(left, right):
    return os.path.join(left, right)


def create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

'''
sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
sample_submission.csv - a sample submission file in the correct format.
items.csv - supplemental information about the items/products.
item_categories.csv  - supplemental information about the items categories.
shops.csv- supplemental information about the shops.
'''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'dataset/future_sales')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'dataset/processed')

TRAIN_FILE = path_join(DATA_DIR, 'sales_train.csv')

TEST_FILE = path_join(DATA_DIR, 'test.csv')
SAMPLE_SUBMISSION_FILE = path_join(DATA_DIR, 'sample_submission.csv')

ITEMS_FILE = path_join(DATA_DIR, 'items.csv')
ITEM_CATEGORIES_FILE = path_join(DATA_DIR, 'item_categories.csv')
SHOPS_FILE = path_join(DATA_DIR, 'shops.csv')

EXPERIMENTS_DIR = path_join(ROOT_DIR, 'experiments')
create_directory(EXPERIMENTS_DIR)


def generate_batch(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


def create_experiment_dirs(experiment_name):
    experiment_dir = os.path.join('.', EXPERIMENTS_DIR, experiment_name)
    create_directory(experiment_dir)

    # experiment_data_dir = os.path.join(experiment_dir, 'dataset')
    # create_directory(experiment_data_file)

    data_file = os.path.join(experiment_dir, 'dataset.csv')
    model_file = os.path.join(experiment_dir, 'model')
    result_file = os.path.join(experiment_dir, 'results.json')

    return experiment_dir, data_file, model_file, result_file
