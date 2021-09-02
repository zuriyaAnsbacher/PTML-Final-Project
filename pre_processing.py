import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from math import isnan
from scipy.stats import zscore
from statistics import stdev
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from ml_algorithm.random_forest import random_forest_algorithm
from ml_algorithm.svm import svm_algorithm
from ml_algorithm.xgboost_run import xgboost_algorithm, xgboost_test, auc_validation_test, save_model_to_pickle

CUR_YEAR = 2019


def write2csv(input_file, output_file, num_line):
    output_file = open(output_file, 'w', newline='')
    writer = csv.writer(output_file)
    i = 0
    with open(input_file) as input_f:
        reader = csv.reader(input_f, delimiter=',')
        for line in reader:
            if i > num_line:
                break
            writer.writerow(line)
            i += 1
    output_file.close()


def blanks_in_sample(df, row, columns_to_remove):
    for column in df.columns:
        if column not in columns_to_remove and isnan(df[column][row]):
            return True
    return False


def pre_split_train_test(df_x, df_y, out_tr_x, out_tr_y, out_ts_x, out_ts_y):
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2)

    train_x.to_csv(out_tr_x)
    test_x.to_csv(out_ts_x)

    train_y.to_csv(out_tr_y)
    test_y.to_csv(out_ts_y)


def pre_processing(file_path):
    df = pd.read_csv(file_path, index_col=0)

    df['sex'] = df['sex'].map({'F': 0, 'M': 1})  # convert F to 0, M to 1

    # # create one-hot vector to F,M
    # df_sex = pd.get_dummies(df['sex'], prefix='sex')
    # if 'sex_ ' in list(df_sex.columns):
    #     df_sex.drop(['sex_ '], axis=1, inplace=True)
    # df = pd.concat([df_sex, df], axis=1)
    # print("done create df")

    # columns_to_remove = ['sex', 'age_at_recruitment', 'reg_year', 'dc', 'dc_cat', 'donate_year',
    #                      'days_until_first_donation', 'days_until_first_request']

    columns_to_remove = ['age_at_recruitment', 'reg_year', 'dc', 'dc_cat', 'donate_year',
                         'days_until_first_donation', 'days_until_first_request']

    CUR_YEAR_COL = np.array([CUR_YEAR] * len(df.index))
    df['cur_age'] = CUR_YEAR_COL - df['reg_year'] + df['age_at_recruitment']
    df['year_from_reg'] = CUR_YEAR_COL - df['reg_year']

    print("before new: " + str(len(df.index)))
    new_df = df[(df['donated'] == 0) | (df['donate_year'] >= CUR_YEAR)]
    new_df = new_df[new_df['cur_age'] <= 60]
    new_df = new_df[new_df['reg_year'] <= CUR_YEAR]

    print("after new: " + str(len(new_df.index)))

    new_df.drop(columns_to_remove, axis=1, inplace=True)  # drop the irrelevant columns

    new_df.dropna(inplace=True)    # drop rows with NaN values
    new_df['donor_mass_log'] = np.log2(new_df['donor_mass'])  # create new feature: log of donor_mass

    df_y = pd.DataFrame(new_df['donated'])
    df_x = new_df.drop(['donated'], axis=1)

    return df_x, df_y


def pre_processing_version2(file_path):
    df = pd.read_csv(file_path, index_col=0)

    df['sex'] = df['sex'].map({'F': 0, 'M': 1})  # convert F to 0, M to 1

    columns_to_remove = ['age_at_recruitment', 'reg_year', 'dc', 'dc_cat', 'donate_year',
                         'days_until_first_donation', 'days_until_first_request']

    CUR_YEAR_COL = np.array([CUR_YEAR] * len(df.index))
    df['cur_age'] = CUR_YEAR_COL - df['reg_year'] + df['age_at_recruitment']
    df['years_from_reg'] = CUR_YEAR_COL - df['reg_year']


    print("before new: " + str(len(df.index)))
    new_df = df[(df['donated'] == 0) | (df['donate_year'] >= CUR_YEAR)]
    new_df = new_df[new_df['cur_age'] <= 60]
    new_df = new_df[new_df['cur_age'] >= 17]
    new_df = new_df[new_df['reg_year'] <= CUR_YEAR]

    print("after new: " + str(len(new_df.index)))

    new_df.drop(columns_to_remove, axis=1, inplace=True)  # drop the irrelevant columns

    new_df.dropna(inplace=True)  # drop rows with NaN values
    # new_df['donor_mass_log'] = np.log2(new_df['donor_mass'])  # create new feature: log of donor_mass

    # print('min value in row cur_age:' + str(new_df['cur_age'].min()))
    # print('max value in row cur_age:' + str(new_df['cur_age'].max()))

    new_df.drop(['cur_age'], axis=1, inplace=True)  # drop the irrelevant columns

    df_y = pd.DataFrame(new_df['donated'])
    df_x = new_df.drop(['donated'], axis=1)

    return df_x, df_y


# tmp function
def check_r_and_s(df_x):
    df_x = pd.read_csv(df_x, index_col=0)

    r, s, cur_age = df_x['r'].values[:100], df_x['s'].values[:100], df_x['cur_age'].values[:100]

    ind_sort = cur_age.argsort()
    r_sort = r[ind_sort]
    s_sort = s[ind_sort]
    cur_age_sort = cur_age[ind_sort]

    plt.plot(cur_age_sort, r_sort, 'bo-', label='r')
    plt.plot(cur_age_sort, s_sort, 'go-', label='s')

    plt.xlabel('cur_age')
    plt.legend()
    plt.show()


def main(input_file):
    df_x, df_y = pre_processing(input_file)
    print("done pre-processing")
    # df_x = z_score_normalization(df_x)
    # print("done zscore")

    return df_x, df_y
    # xgboost_algorithm(df_x, df_y)


def main_after_process(train_x_f, train_y_f):
    df_x = pd.read_csv(train_x_f, index_col=0)
    df_y = pd.read_csv(train_y_f, index_col=0)

    xgboost_algorithm(df_x, df_y, None)


def main_test(test_x, test_y):
    df_x = pd.read_csv(test_x, index_col=0)
    df_y = pd.read_csv(test_y, index_col=0)

    xgboost_test(df_x, df_y)


def main_create_auc(tr_x, tr_y, ts_x, ts_y):
    df_x_tr = pd.read_csv(tr_x, index_col=0)
    df_y_tr = pd.read_csv(tr_y, index_col=0)
    df_x_ts = pd.read_csv(ts_x, index_col=0)
    df_y_ts = pd.read_csv(ts_y, index_col=0)

    auc_validation_test(df_x_tr, df_y_tr, df_x_ts, df_y_ts)


def main_load_model(tr_x, tr_y, ts_x):
    df_x_tr = pd.read_csv(tr_x, index_col=0)
    df_y_tr = pd.read_csv(tr_y, index_col=0)
    df_x_ts = pd.read_csv(ts_x, index_col=0)

    save_model_to_pickle(df_x_tr, df_y_tr, df_x_ts, 'save model/xgboost_model_2019.pkl')


# main_create_auc('split_new1/train_x_2019.csv', 'split_new1/train_y_2019.csv',
#                 'split_new1/test_x_2019.csv',  'split_new1/test_y_2019.csv')

# print("*********** 2019 ***********")
# main_create_auc('split_new1/train_x_2019.csv', 'split_new1/train_y_2019.csv',
#                 'split_new1/test_x_2019.csv',  'split_new1/test_y_2019.csv')

# df_x, df_y = pre_processing_version2('input/anon.2020-10-19.csv')
# pre_split_train_test(df_x, df_y, 'split with change features/train_X_2019.csv',
#                      'split with change features/train_Y_2019.csv',
#                      'split with change features/test_X_2019.csv',
#                      'split with change features/test_Y_2019.csv')

main_load_model('split with change features/train_X_2019.csv',
                'split with change features/train_Y_2019.csv',
                'split with change features/test_X_2019.csv')
