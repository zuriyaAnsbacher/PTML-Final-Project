import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.axes import Axes
import seaborn as sns

def calculate_pos_neg(df_y_train, df_y_test):
    y_train = pd.read_csv(df_y_train, index_col=0)
    y_test = pd.read_csv(df_y_test, index_col=0)

    print('train:')
    print(y_train['donated'].value_counts())
    print('test:')
    print(y_test['donated'].value_counts())


# create log scale between the minimum and maximum of f, with bins = num_bins
def create_log_scale(f, num_bins, epsilon):
    hist_f, bins_f = np.histogram(f, bins=num_bins)
    # can not calculate log(0) so calculate log(epsilon), create log scale, and then switch the first element with 0
    if bins_f[0] == 0:
        logbins_f = np.logspace(np.log10(epsilon), np.log10(bins_f[-1]), len(bins_f))
        logbins_f[0] = 0
    else:
        logbins_f = np.logspace(np.log10(bins_f[0]), np.log10(bins_f[-1]), len(bins_f))
    return logbins_f


def histogram_plot(f1, f2, f3, f4, feature_name):
    epsilon = 0.00001
    num_bins = 20
    f_for_bins = f1
    plt.rcParams["font.family"] = "Times New Roman"
    colors_ = ['#888888', '#E6194B', '#88CCEE', '#332288']
    plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 200})


    if feature_name == 'sex':
        feature_name = 'gender'

    if feature_name in ['s', 'lambda', 'donor_mass', 'year_from_reg']:  # log scale (x)
        bins_ = create_log_scale(f_for_bins, num_bins, epsilon)
        flag = 0
    elif feature_name in ['gender', 'r', 'cur_age', 'donor_mass_log']:  # linear scale (x)
        _, bins_ = np.histogram(f_for_bins, bins=num_bins)
        flag = 1
    else:
        print('--------- error ---------')
        exit()

    plt.hist([f1, f2, f3, f4], bins=bins_, align='left', label=['2016', '2017', '2018', '2019'],
             color=colors_, rwidth=0.6)
    plt.ylabel('Frequency', fontsize=20)  # fontdict?
    plt.legend(prop={'size': 15})

    if flag == 0:
        plt.xscale('log')
    plt.xticks(fontsize=15)

    plt.savefig(f'histogram fig/{feature_name} : Histogram.png')
    plt.show()


# def calculate_histogram(f1, f2, f3, f4, feat_name):
#     with open('histogram/features histogram.txt', 'a+') as file:
#         if feat_name in ['sex', 'r', 'cur_age', 'donor_mass_log']:  # linear scale
#             for ind, f in enumerate([f1, f2, f3, f4]):
#                 hist_, bins_ = np.histogram(f, bins=10)
#                 file.write(f'\n year: {ind} feature:{feat_name}\n')
#                 file.write(','.join(str(item) for item in list(hist_)))
#             print("bins linear")
#             print(bins_)
#
#         elif feat_name in ['s', 'lambda', 'donor_mass', 'year_from_reg']:  #log scale
#             for ind, f in enumerate([f1, f2, f3, f4]):
#                 hist_demo, bins_demo = np.histogram(f, bins=10)
#                 if bins_demo[0] == 0:
#                     log_bins = np.logspace(0, bins_demo[-1], 11) / 10
#                     log_bins[0] = 0
#                     # add
#                 else:
#                     log_bins = np.logspace(np.log10(bins_demo[0]), np.log10(bins_demo[-1]), 11)
#                     # add
#                 file.write(f'\n year: {ind} feature:{feat_name}\n')
#                 file.write(','.join(str(item) for item in list(hist_)))
#             print("bins log")
#             print(bins_demo)
#         else:
#             print('error')
#             exit()


def df2feature(df1_tr, df1_ts, df2_tr, df2_ts, df3_tr, df3_ts, df4_tr, df4_ts):
    df1_tr = pd.read_csv(df1_tr, index_col=0)
    df1_ts = pd.read_csv(df1_ts, index_col=0)

    df1 = pd.concat([df1_tr, df1_ts])

    df2_tr = pd.read_csv(df2_tr, index_col=0)
    df2_ts = pd.read_csv(df2_ts, index_col=0)

    df2 = pd.concat([df2_tr, df2_ts])

    df3_tr = pd.read_csv(df3_tr, index_col=0)
    df3_ts = pd.read_csv(df3_ts, index_col=0)

    df3 = pd.concat([df3_tr, df3_ts])

    df4_tr = pd.read_csv(df4_tr, index_col=0)
    df4_ts = pd.read_csv(df4_ts, index_col=0)

    df4 = pd.concat([df4_tr, df4_ts])

    for col in df1.columns:
        histogram_plot(df1[col], df2[col], df3[col], df4[col], col)


df2feature("split_new1/train_x_2016.csv", "split_new1/test_x_2016.csv", "split_new1/train_x_2017.csv",
           "split_new1/test_x_2017.csv", "split_new1/train_x_2018.csv", "split_new1/test_x_2018.csv",
           "split_new1/train_x_2019.csv", "split_new1/test_x_2019.csv",)

#e6194b
#3cb44b
#0082c8
#f58231
#e6194b
#911eb4
#46f0f0
#f032e6
#d2f53c
#fabebe
#008080
#e6beff
#aa6e28
#800000
