import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from definitions import *


def sum_by_months(df):
    df['date'] = pd.to_datetime(df['date'])

    # perform GroupBy operation over monthly frequency
    res = df.set_index('date').groupby(['item_id', pd.Grouper(freq='M')])['item_cnt_day'].sum().reset_index()

    print(res[res['item_id'] == 1])


def add_columns(data):
    data.groupby(['item_id'])


def sales_file_processing():
    data = pd.read_csv(TRAIN_FILE)
    data.loc[data.shop_id == 0, 'shop_id'] = 57
    data.loc[data.shop_id == 1, 'shop_id'] = 58
    data.loc[data.shop_id == 11, 'shop_id'] = 10
    data.loc[data.shop_id == 40, 'shop_id'] = 39

    # sum_by_months(data)
    # # data['date'] = pd.to_datetime(data['date'])
    # # print(data.groupby([data['date'].dt.strftime('%B'), data['item_id']])['item_cnt_day'].sum())
    return data


# sales_df = sales_file_processing()
# test_df = pd.read_csv(TEST_FILE)
#
# train_shops = set(sales_df['shop_id'].unique())
# test_shops = set(test_df['shop_id'].unique())
# print("train_shops {}: ".format(len(train_shops)), train_shops)
# print("test_shops {}: ".format(len(test_shops)), test_shops)
# intersection = train_shops.intersection(test_shops)
# print("train intersection test {}: ".format(len(intersection)), intersection)
#
#
# train_items = set(sales_df['item_id'].unique())
# test_items = set(test_df['item_id'].unique())
# print("train_items {}: ".format(len(train_items)), train_items)
# print("test_items {}: ".format(len(test_items)), test_items)
# intersection = train_items.intersection(test_items)
# print("train intersection test {}: ".format(len(intersection)), intersection)


def read_dataset(input_file):
    df = pd.read_csv(input_file)
    df = df.fillna(0)

    df['ID'] = df['ID'].astype(np.uint32)
    df['item_id'] = df['item_id'].astype(np.uint16)
    for column in df.columns:
        _dtype = df[column].dtype
        if _dtype == 'int64':
            df[column] = df[column].astype(np.uint8)
        if _dtype == 'float64':
            df[column] = df[column].astype(np.float16)
    # df['shop_id'] = df['shop_id'].astype('category')
    # df['date_block_num'] = df['date_block_num'].astype('category')
    # df['days'] = df['days'].astype('category')
    # df['month'] = df['month'].astype('category')
    # df['season'] = df['season'].astype('category')
    # df['item_category_id'] = df['item_category_id'].astype('category')
    # df['city_id'] = df['city_id'].astype('category')
    # df['category_id'] = df['category_id'].astype('category')
    # df['subcategory_id'] = df['subcategory_id'].astype('category')
    # df['subsubcategory_id'] = df['subsubcategory_id'].astype('category')

    return df


def shops_file_processing():
    shops_df = pd.read_csv(SHOPS_FILE)
    shops_df['city'] = shops_df['shop_name'].apply(lambda x: x.split(' ')[0])
    # shops_df = pd.concat([shops_df, pd.get_dummies(shops_df['city'], prefix='city')], axis=1)
    shops_df['city_id'] = pd.factorize(shops_df['city'])[0]
    shops_df.drop(['city', 'shop_name'], axis=1, inplace=True)

    shops_df['city_id'] = shops_df['city_id'].astype(np.uint8)

    return shops_df


def categories_file_processing():
    items_categories_df = pd.read_csv(ITEM_CATEGORIES_FILE)

    items_categories_df['category'] = items_categories_df['item_category_name'].apply(lambda x: x.split(' - ')[0])
    items_categories_df['subcategory'] = items_categories_df['item_category_name'].apply(lambda x: x.split(' - ')[1])
    items_categories_df['subsubcategory'] = items_categories_df['item_category_name'].apply(lambda x: x.split(' - ')[2])

    # items_categories_df = pd.concat([items_categories_df, pd.get_dummies(items_categories_df['category'], prefix='category')], axis=1)
    # items_categories_df = pd.concat([items_categories_df, pd.get_dummies(items_categories_df['subcategory'], prefix='subcategory')], axis=1)
    # items_categories_df = pd.concat([items_categories_df, pd.get_dummies(items_categories_df['subsubcategory'], prefix='subsubcategory')], axis=1)

    items_categories_df['category_id'] = pd.factorize(items_categories_df['category'])[0]
    items_categories_df['subcategory_id'] = pd.factorize(items_categories_df['subcategory'])[0]
    items_categories_df['subsubcategory_id'] = pd.factorize(items_categories_df['subsubcategory'])[0]

    items_categories_df.drop(['category', 'subcategory', 'subsubcategory', 'item_category_name'], axis=1, inplace=True)

    items_categories_df['category_id'] = items_categories_df['category_id'].astype(np.uint8)
    items_categories_df['subcategory_id'] = items_categories_df['subcategory_id'].astype(np.uint8)
    items_categories_df['subsubcategory_id'] = items_categories_df['subsubcategory_id'].astype(np.uint8)

    return items_categories_df


def test_file_processing():
    data = pd.read_csv(TEST_FILE)
    data.loc[data.shop_id == 0, 'shop_id'] = 57
    data.loc[data.shop_id == 1, 'shop_id'] = 58
    data.loc[data.shop_id == 11, 'shop_id'] = 10
    data.loc[data.shop_id == 40, 'shop_id'] = 39

    data['date_block_num'] = 34
    data['date_block_num'] = data['date_block_num'].astype(np.uint8)
    data['shop_id'] = data['shop_id'].astype(np.uint8)
    data['item_id'] = data['item_id'].astype(np.uint16)
    return data


# print(categories_file_processing().head(10))

# print(test_df['shop_id'].unique().shape[0])
# shops_file_processing()



# print(sale_train.shape)
# print("----------Top-5- Record----------")
# print(sale_train.head(5))
# print("-----------Information-----------")
# print(sale_train.describe())
# # print("----------Missing value-----------")
# # print(sale_train.isnull().sum())
# # print("----------Null value-----------")
# # print(sale_train.isna().sum())
# test = pd.read_csv(test_file)
# print(test.shape)
# print("----------Top-5- Record----------")
# print(test.head(5))
# print("-----------Information-----------")
# print(test.describe())
#
# test = pd.read_csv(sample_submission_file)
# print(test.shape)
# print("----------Top-5- Record----------")
# print(test.head(5))
# print("-----------Information-----------")
# print(test.describe())


# test = pd.read_csv(items_file)
# print(test.shape)
# print("----------Top-5- Record----------")
# print(test.head(5))
# print("-----------Information-----------")
# print(test.info())
# print(test.describe())

# test = pd.read_csv(item_categories_file)
# print(test.shape)
# print("----------Top-5- Record----------")
# print(test)
# print("-----------Information-----------")
# print(test.info())
# print(test.describe())


