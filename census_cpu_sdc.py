#!/usr/bin/env python
# coding: utf-8
import argparse
import gzip
import os
import sys
import warnings
from timeit import default_timer as timer

import numpy as np
import numba
import time

import pandas as pd

from numba import literal_unroll

from sklearn.model_selection import train_test_split
import sdc
# import vtune as vt
from vtune import task_begin, task_end, domain, string_handle_create
# import ctypes
# import itt


handle = string_handle_create("Function")
handle_main = string_handle_create("Main")
handle_func = string_handle_create("Func_ALL")

# warnings.filterwarnings('ignore')

# Dataset link
# https://rapidsai-data.s3.us-east-2.amazonaws.com/datasets/ipums_education2income_1970-2010.csv.gz 


# Original loading
def load_data(cached = 'ipums_education2income_1970-2010.csv.gz', source='ipums'):
    if os.path.exists(cached) and source=='ipums':
        with gzip.open(cached) as f:
            X = pd.read_csv(f)
    else:
        print("No data found!  Please uncomment the cell above the 'LOAD_DATA' function and try again!")
        X = None
    return X


def numba_jit(*args, **kwargs):
    kwargs.update({'parallel': True, 'nopython': True})
    return numba.njit(*args, **kwargs)


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()

sdc_mse = numba_jit(mse)

def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ( (y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)

sdc_cod = numba_jit(cod)


def run_etl_pandas(df):
# def run_etl_pandas(file_name, df):
    task_begin(domain, handle)
    dataset_path = "C:\\Users\\etotmeni\\sdc\\usa_00009.csv"

    # Read only this cols
    keep_cols = ['YEAR', 'DATANUM', 'SERIAL', 'CBSERIAL', 'HHWT', 'GQ', 'PERNUM', 'SEX', 'AGE', 'INCTOT', 'EDUC', 'EDUCD', 'EDUC_HEAD', 'EDUC_POP', 'EDUC_MOM','EDUCD_MOM2','EDUCD_POP2', 
    'INCTOT_MOM','INCTOT_POP','INCTOT_MOM2','INCTOT_POP2', 'INCTOT_HEAD', 'SEX_HEAD', 'CPI99']

    dtypes = {'YEAR': np.float64, 
    'DATANUM': np.float64, 
    'SERIAL': np.float64, 
    'CBSERIAL': np.float64, 
    'HHWT': np.float64, 
    'GQ': np.float64, 
    'PERNUM': np.float64, 
    'SEX': np.float64, 
    'AGE': np.float64, 
    'INCTOT': np.float64, 
    'EDUC': np.float64, 
    'EDUCD': np.float64, 
    'EDUC_HEAD': np.float64, 
    'EDUC_POP': np.float64, 
    'EDUC_MOM': np.float64,
    'EDUCD_MOM2': np.float64,
    'EDUCD_POP2': np.float64, 
    'INCTOT_MOM': np.float64,
    'INCTOT_POP': np.float64,
    'INCTOT_MOM2': np.float64,
    'INCTOT_POP2': np.float64, 
    'INCTOT_HEAD': np.float64, 
    'SEX_HEAD': np.float64, 
    'CPI99': np.float64}

    # # start = time.time()
    df = pd.read_csv(numba.literally(dataset_path), usecols=keep_cols, dtype=dtypes)
    # print(df)
    # t_readcsv = time.time() - start
    # print(df.head(5))
    # t0 = time.time()
    mask = df['INCTOT'] != 9999999
    # t_mask = time.time() - t0 

    # t0 = time.time()
    df = df[mask]
    # t_bool_getitem = time.time() - t0

    # t0 = time.time()
    df = df.reset_index(drop=True)
    # t_reset_index = time.time() - t0


    # t0 = time.time()
    res = df['INCTOT'] * df['CPI99']
    df = df._set_column('INCTOT', res) #  sdc
    # df['INCTOT'] = res #  python
    # t_arithm = time.time() - t0

    # t0 = time.time()
    mask1 = df['EDUC'].notna()
    # t_mask += time.time() - t0

    # t0 = time.time()
    df = df[mask1]
    # t_bool_getitem += time.time() - t0

    # t0 = time.time()
    df = df.reset_index(drop=True)
    # t_reset_index = time.time() - t0

    # t0= time.time()
    mask2 = df['EDUCD'].notna()
    # t_mask += time.time() - t0

    # t0 = time.time()
    df = df[mask2]
    # t_bool_getitem += time.time() - t0

    # t0 = time.time()
    df = df.reset_index(drop=True)
    # t_reset_index = time.time() - t0

    # t0 = time.time()
    df['YEAR'].fillna(-1, inplace=True)
    # t_fillna_inplace_one = time.time() - t0

    # t0 = time.time()
    df['DATANUM'].fillna(-1, inplace=True)
    df['SERIAL'].fillna(-1, inplace=True)
    df['CBSERIAL'].fillna(-1, inplace=True)
    df['HHWT'].fillna(-1, inplace=True)
    df['GQ'].fillna(-1, inplace=True)
    df['PERNUM'].fillna(-1, inplace=True)
    df['SEX'].fillna(-1, inplace=True)
    df['AGE'].fillna(-1, inplace=True)
    df['INCTOT'].fillna(-1, inplace=True)
    df['EDUCD'].fillna(-1, inplace=True)
    df['EDUC_HEAD'].fillna(-1, inplace=True)
    df['INCTOT_HEAD'].fillna(-1, inplace=True)
    df['EDUC_POP'].fillna(-1, inplace=True)
    df['EDUC_MOM'].fillna(-1, inplace=True)
    df['EDUCD_MOM2'].fillna(-1, inplace=True)
    df['EDUCD_POP2'].fillna(-1, inplace=True)
    df['INCTOT_MOM'].fillna(-1, inplace=True)
    df['INCTOT_POP'].fillna(-1, inplace=True)
    df['INCTOT_MOM2'].fillna(-1, inplace=True)
    df['INCTOT_POP2'].fillna(-1, inplace=True)
    df['SEX_HEAD'].fillna(-1, inplace=True)

    # t_fillna_inplace_all = time.time() - t0

    # t0 = time.time()
    y = df["EDUC"]
    # t_getitem = time.time() - t0

    # t0 = time.time()
    X = df.drop(columns="EDUC")
    # t_drop = time.time() - t0

    # end_time = time.time() - start
    # end_time = 0

    # time_results = [t_arithm, t_readcsv, t_reset_index, t_mask, t_bool_getitem,  t_fillna_inplace_one, t_fillna_inplace_all, t_drop, t_getitem]
    # time_results = [t_arithm, t_reset_index, t_mask, t_bool_getitem,  t_fillna_inplace_one, t_fillna_inplace_all, t_drop, t_getitem]
    task_end(domain)
    # return X, y, end_time, time_results
    return X, y
    # return 0, 0


sdc_etl_pandas = numba_jit(run_etl_pandas)


def train_and_test(X, y, clf):
    t_train_test_split = 0.0
    t_train = 0.0
    t_inference = 0.0
    
    mse_values, cod_values = [], []
    N_RUNS = 50
    TRAIN_SIZE = 0.9
    random_state = 777

    for i in range(N_RUNS):
        t0 = timer()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=random_state, shuffle=False)
        t_train_test_split += timer() - t0
        random_state += 777

        t0 = timer()
        model = clf.fit(X_train, y_train)
        t_train += timer() - t0

        t0 = timer()
        y_pred = model.predict(X_test)
        t_inference += timer() - t0

        mse_values.append(mse(y_test, y_pred))
        cod_values.append(cod(y_test, y_pred))

    return mse_values, cod_values, t_train_test_split, t_train, t_inference


def sdc_pipeline(dataset_path, clf):
    print("SDC pipeline")
    times_results = {}
    print("Time in seconds")

    func_titles = ["t_arithm", "t_csv", "t_reset_index", "t_mask", "t_bool_getitem", "t_fillna_inplace_one", "t_fillna_inplace_all",
                   "t_drop", "t_getitem"]

    # ETL PART

    # dataset_path = 'ipums_education2income_1970-2010.csv'
    # Read only this cols
    # keep_cols = ['YEAR', 'DATANUM', 'SERIAL', 'CBSERIAL', 'HHWT', 'GQ', 'PERNUM', 'SEX', 'AGE', 'INCTOT', 'EDUC', 'EDUCD', 'EDUC_HEAD', 'EDUC_POP', 'EDUC_MOM','EDUCD_MOM2','EDUCD_POP2', 
    # 'INCTOT_MOM','INCTOT_POP','INCTOT_MOM2','INCTOT_POP2', 'INCTOT_HEAD', 'SEX_HEAD', 'CPI99']

    # dtypes = {'YEAR': np.float64, 
    # 'DATANUM': np.float64, 
    # 'SERIAL': np.float64, 
    # 'CBSERIAL': np.float64, 
    # 'HHWT': np.float64, 
    # 'GQ': np.float64, 
    # 'PERNUM': np.float64, 
    # 'SEX': np.float64, 
    # 'AGE': np.float64, 
    # 'INCTOT': np.float64, 
    # 'EDUC': np.float64, 
    # 'EDUCD': np.float64, 
    # 'EDUC_HEAD': np.float64, 
    # 'EDUC_POP': np.float64, 
    # 'EDUC_MOM': np.float64,
    # 'EDUCD_MOM2': np.float64,
    # 'EDUCD_POP2': np.float64, 
    # 'INCTOT_MOM': np.float64,
    # 'INCTOT_POP': np.float64,
    # 'INCTOT_MOM2': np.float64,
    # 'INCTOT_POP2': np.float64, 
    # 'INCTOT_HEAD': np.float64, 
    # 'SEX_HEAD': np.float64, 
    # 'CPI99': np.float64}

    # start = time.time()
    # df = pd.read_csv(dataset_path, usecols=keep_cols, dtype=dtypes)



    # df = pd.read_csv(dataset_path)
    # print(df.head())
    # t_readcsv = time.time() - start
    # print(type(df))
    # warm_up_time = time.time()
    # X, y = sdc_etl_pandas(dataset_path, df)
    df = 0
    X, y = sdc_etl_pandas(df)
    # dftest = pd.DataFrame({'A': [1., 2., 4., 6., 4., 2.], 'B': [3., 2., 77., 2., 5., 6.5]})
    task_begin(domain, handle_func)
    X, y = sdc_etl_pandas(df)
    task_end(domain)
    # wu_time = time.time() - warm_up_time
    # print("SDC WARMUP = ", wu_time)

    # # start = time.time()
    # X, y = sdc_etl_pandas(dataset_path, df)
    # # t_etl = time.time() - start

    # # for name, t in zip(func_titles, func_times):
    # #     print(f"SDC {name} time: {t}")

    # # ML PART
    
    # mse_values, cod_values, t_train_test_split, t_train, t_inference = train_and_test(X, y, clf)

    # # print('t_ETL = ', t_etl)
    # print('t_train_test_split = ', t_train_test_split)
    # print('t_ML = ', t_train + t_inference)
    # print('  t_train = ', t_train)
    # print('  t_inference = ', t_inference)
    # mean_mse = sum(mse_values)/len(mse_values)
    # mean_cod = sum(cod_values)/len(cod_values)
    # mse_dev = pow(sum([(mse_value - mean_mse)**2 for mse_value in mse_values])/(len(mse_values) - 1), 0.5)
    # cod_dev = pow(sum([(cod_value - mean_cod)**2 for cod_value in cod_values])/(len(cod_values) - 1), 0.5)
    # print("\nmean MSE ± deviation: {:.9f} ± {:.9f}".format(mean_mse, mse_dev))
    # print("mean COD ± deviation: {:.9f} ± {:.9f}".format(mean_cod, cod_dev))


# def python_pipeline(dataset_path, clf):
#     print("STOVK Python pipeline")
#     times_results = {}
#     print("Time in seconds")

#     func_titles = ["t_arithm", "t_csv", "t_reset_index", "t_mask", "t_bool_getitem", "t_fillna_inplace_one", "t_fillna_inplace_all",
#                    "t_drop", "t_getitem"]

#     # ETL PART
#     start = time.time()
#     # X, y, nobu_time, func_times = run_etl_pandas(dataset_path, df)
#     X, y = run_etl_pandas(dataset_path, df)
#     t_etl = time.time() - start

#     print("ETL TIME: ", t_etl)

#     # for name, t in zip(func_titles, func_times):
#     #     print(f"{name} time: {t}")

#     # print("FUNCT_TIMES: ", func_times)

#     # ML PART
#     mse_values, cod_values, t_train_test_split, t_train, t_inference = train_and_test(X, y, clf)

#     end = time.time() - start

#     print('t_train_test_split = ', t_train_test_split)
#     print('t_ML = ', t_train + t_inference)
#     print('  t_train = ', t_train)
#     print('  t_inference = ', t_inference)

#     print('TOTAL = ', end)
#     mean_mse = sum(mse_values)/len(mse_values)
#     mean_cod = sum(cod_values)/len(cod_values)
#     mse_dev = pow(sum([(mse_value - mean_mse)**2 for mse_value in mse_values])/(len(mse_values) - 1), 0.5)
#     cod_dev = pow(sum([(cod_value - mean_cod)**2 for cod_value in cod_values])/(len(cod_values) - 1), 0.5)
#     print("\nmean MSE ± deviation: {:.9f} ± {:.9f}".format(mean_mse, mse_dev))
#     print("mean COD ± deviation: {:.9f} ± {:.9f}".format(mean_cod, cod_dev))


if __name__ == '__main__':
    task_begin(domain, handle_main)
    # dataset_path = "C:\\Users\\etotmeni\\sdc\\ipums_education2income_1970-2010.csv"
    dataset_path = "C:\\Users\\etotmeni\\sdc\\usa_00009.csv"
    NUM_THREADS = os.environ.get('NUMBA_NUM_THREADS', 1)

    parser = argparse.ArgumentParser()

    parser.add_argument("--daal4py", help="switch to speeded up scikit-learn",
                        action="store_true")
    parser.add_argument("--sdc", help="switch to sdc pipeline",
                        action="store_true")
    args = parser.parse_args()

    if args.daal4py:
        import daal4py
        from daal4py import sklearn, daalinit
        daalinit(int(NUM_THREADS))

        print("Intel optimized sklearn is used")
        clf = daal4py.sklearn.linear_model.Ridge()

    else:
        import sklearn
        print("Stock sklearn is used")
        import sklearn.linear_model as lm
        clf = lm.Ridge()

    if args.sdc:
        sdc_pipeline(dataset_path, clf)
    # else:
    #     python_pipeline(dataset_path, clf)
    task_end(domain)