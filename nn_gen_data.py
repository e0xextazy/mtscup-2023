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
import tqdm
import torch
import pickle
import joblib


def reduce_mem_usage(df):
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


def get_words_dict(texts_list):
    words_set = set([])
    for text in texts_list:
        words_set = words_set.union(str(text).split())
        
    words_dict = {}
    for i, word in enumerate(words_set):
        words_dict[word] = i+1
        
    return words_dict


def get_tensor(text, words_dict, top_n=5):
    text_split = str(text).split()
    token_indx = []
    for word in text_split:
        token_indx.append(words_dict.get(word, 0))
        
    token_indx = token_indx[:top_n]
    if len(token_indx) < top_n:
        token_indx += [0]*(top_n - len(token_indx))
        
    return token_indx


with open('preprocessor_nn.p', 'rb') as f:
    preprocessor = pickle.load(f)
# urls_texts = pd.read_csv("vanya_automl/urls_texts_new (5).csv")
# urls_texts.drop("Unnamed: 0", axis=1, inplace=True)
with open('words_dict.pickle', 'rb') as handle:
    words_dict = pickle.load(handle)
with open("vanya_automl/replace_urls (2).pickle", "rb") as f:
    replace_urls = pickle.load(f)
    
    
urls_texts = pd.read_csv("vanya_automl/urls_texts_new (6).csv")[["url_host", "sum_text", "urls_topics"]]
urls_texts['urls_topics'] = urls_texts['urls_topics'].apply(lambda x: str(x).split("_")[0] + "_bt") 
pqt_path = "competition_data_final_pqt"
parquets = [parquet for parquet in os.listdir(pqt_path) if "parquet" in parquet]#[:1]
data = []
for parquet in tqdm.tqdm(parquets):
    df = reduce_mem_usage(pd.read_parquet(f"{pqt_path}/{parquet}"))#[:1000]
    df.sort_values(['user_id', 'date', 'part_of_day'], inplace=True)
    
    df['price'].fillna(1., inplace=True)
    df['price'] = df['price'].apply(lambda x: np.abs(x/1000) + 1)
    df.price = df.price.astype('float16')
    
    df["url_host"] = df["url_host"].apply(lambda x: replace_urls.get(x, x))
    df = df.merge(urls_texts[["url_host", "urls_topics"]], how="left", left_on="url_host", right_on="url_host")
    df["urls_topics"].fillna("nt", inplace=True)
    df["urls_topics"] = df["urls_topics"].astype('category')
    
    data_buf = preprocessor.transform(df)
    data_buf.sort(key=lambda x: x['user_id'])
    df.drop(["urls_topics", "region_name", "city_name"] , axis=1, inplace=True)
         
    df = df[["user_id", "url_host"]].merge(urls_texts[["url_host", "sum_text"]], how='left', on='url_host')
    df = df.groupby("user_id", as_index=False).agg(list)
    df.sort_values("user_id", inplace=True)
    for i in range(len(df)):
        data_buf[i]["text"] = torch.LongTensor([get_tensor(text, words_dict) for text in df.iloc[i]['sum_text']])
    data += data_buf
    
    del df
    del data_buf
    gc.collect()
        
data = sorted(data, key=lambda x: x['user_id'])
def fastdump(obj, file):
    p = pickle.Pickler(file)
    p.fast = True
    p.dump(obj)
    
with open('data_nn_wt.pickle', 'wb') as handle:
    fastdump(data, handle)