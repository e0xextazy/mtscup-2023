# !pip install git+https://github.com/livington/pytorch-lifestream.git@main >> none

import sys
import os
import gc
import warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import time
import pyarrow.parquet as pq
import pyarrow as pa
import scipy
import implicit
import bisect
import sklearn.metrics as m
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import pickle
from ptls.preprocessing import PandasDataPreprocessor


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


pqt_path = '../competition_data_final_pqt'
with open("replace_urls (2).pickle", "rb") as f:
    replace_urls = pickle.load(f)
urls_texts = pd.read_csv("urls_texts_new (5).csv")[["url_host", "urls_topics"]]
urls_texts['urls_topics'] = urls_texts['urls_topics'].apply(lambda x: str(x).split("_")[0] + "_bt") 
data = pd.concat([
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00000-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')),
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00001-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')),
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00002-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')),
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00003-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')),
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00004-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')),
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00005-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')),  
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00006-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')),
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00007-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')),
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00008-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet')),
                  reduce_mem_usage(pd.read_parquet(f'{pqt_path}/part-00009-aba60f69-2b63-4cc1-95ca-542598094698-c000.snappy.parquet'))
])
data.sort_values(["user_id", "date", "part_of_day"], inplace=True)

data["url_host"] = data["url_host"].apply(lambda x: replace_urls.get(x, x))
data = data.merge(urls_texts, how="left", left_on="url_host", right_on="url_host")
data["urls_topics"].fillna("nt", inplace=True)
data['price'].fillna(1., inplace=True)
data['price'] = data['price'].apply(lambda x: np.abs(x/1000) + 1)


preprocessor = PandasDataPreprocessor(
    col_id='user_id',
    col_event_time='date',
    cols_category=['region_name', "city_name", "cpe_manufacturer_name", 'cpe_model_name', 'url_host',
                   'cpe_type_cd', "cpe_model_os_type", 'part_of_day', 'request_cnt', "urls_topics"],
    cols_numerical=['price'],
    return_records=True,
)
preprocessor.fit(data)
with open('preprocessor_nn.p', 'wb') as f:
    preprocessor = pickle.dump(preprocessor, f)