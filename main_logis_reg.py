import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# evaluate model by auc for train and validation
def evaluate(train_labels, train_probs, test_labels, test_probs):
    train_results = roc_auc_score(train_labels, train_probs)
    results = roc_auc_score(test_labels, test_probs)

    return round(train_results, 5), round(results, 5)


# evaluate model by accuracy
def calc_acc(y_label, y_pred):
    correct_results_sum = sum(y_pred.ravel() == y_label.ravel())
    acc = correct_results_sum / y_label.shape[0]
    acc = (acc * 100)

    return acc


def normalization_zscore(df_x, df_y):
    df_x_without_norm = df_x['donor_mass_log']
    df_x.drop(['donor_mass_log'], axis=1, inplace=True)

    df_x_without_norm, df_x, df_y = df_x_without_norm.values, df_x.values, df_y.values

    df_x = zscore(df_x)
    columns = [df_x[:, i] for i in range(len(df_x[0]))]  # split to columns list
    columns.append(df_x_without_norm)  # add the column without normalization (donor_mass_log)
    columns = [col.reshape(len(col), 1) for col in columns]  # (len, ) to (len, 1)
    return np.concatenate(columns, axis=1), df_y


# logistic regression for nest loops over parameters (find the best)
def logistic_regression(df_x, df_y):

    df_x = pd.read_csv(df_x, index_col=0)
    df_y = pd.read_csv(df_y, index_col=0)

    df_x_without_norm = df_x['donor_mass_log']
    df_x.drop(['donor_mass_log'], axis=1, inplace=True)

    df_x, df_y = normalization_zscore(df_x, df_y)

    # df_x = df_x.values
    # df_y = df_y.values
    # df_x = zscore(df_x)

    # # many parameters (first run)
    # penalty = ['l2', 'l1', 'elasticnet', 'none']
    # dual = [False, True]
    # tol = [0.00001, 0.0001, 0.001]
    # C = [0.7, 1.0, 1.7, 4.0]
    # fit_intercept = [True, False]
    # intercept_scaling = [0.5, 1.0, 1.5]
    # solver = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    # max_iter = [50, 100, 300]
    # verbose = [0.5, 1, 1.5]  # only for solver 'liblinear', 'lbfgs'
    # warm_start = [True, False]

    # # parameters that could be more relevant to our data
    # penalty = ['l2', 'l1']
    # dual = [False, True]
    # tol = [0.000001, 0.0001, 0.01]
    # C = [0.2, 1.0, 1.8]
    # fit_intercept = [True, False]
    # intercept_scaling = [0.2, 1.0, 1.9]
    # solver = ['newton-cg', 'sag', 'saga']
    # max_iter = [20, 100, 350]
    # verbose = [0]  # only for solver 'liblinear', 'lbfgs'
    # warm_start = [True, False]

    penalty = ['l2']
    dual = [False]
    tol = [0.0000001]
    C = [1.0]
    fit_intercept = [True]
    intercept_scaling = [0.2]
    solver = ['newton-cg']
    max_iter = [100]
    verbose = [0]  # only for solver 'liblinear', 'lbfgs'
    warm_start = [False]

    for pen in penalty:
        for du in dual:
            for t in tol:
                for c in C:
                    for fit_int in fit_intercept:
                        for inter in intercept_scaling:
                            for sol in solver:
                                # if sol in ['newton-cg', 'lbfgs', 'sag', 'saga'] and pen == 'l1':
                                #     continue
                                if sol in ['liblinear', 'newton-cg', 'lbfgs', 'sag'] and pen == 'elasticnet':
                                    continue
                                for max_it in max_iter:
                                    for ver in verbose:
                                        if sol == 'newton-cg' and du == True:
                                            continue
                                        # if sol in ['newton-cg', 'sag', 'saga']:
                                        #     continue
                                        for war in warm_start:
                                            try:
                                                x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)
                                                logreg = LogisticRegression(class_weight='balanced', penalty=pen, dual=du,
                                                                            tol=t, C=c, fit_intercept=fit_int,
                                                                            intercept_scaling=inter, solver=sol, max_iter=max_it,
                                                                            verbose=ver, warm_start=war)
                                                logreg.fit(x_train, y_train)
                                                y_pred = logreg.predict(x_test)

                                                acc = round(calc_acc(y_test, y_pred), 5)

                                                y_train_pred1 = logreg.predict_proba(x_train)[:, 1]
                                                y_pred1 = logreg.predict_proba(x_test)[:, 1]

                                                tr_roc1, ts_roc1 = evaluate(y_train.ravel(), y_train_pred1, y_test.ravel(), y_pred1)

                                                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                                print(' '.join(['pen', str(pen), 'du', str(du), 't', str(t), 'c', str(c),
                                                                'fit_int', str(fit_int), 'inter', str(inter), 'sol', str(sol),
                                                                'max_it', str(max_it), 'ver', str(ver), 'war', str(war)]))

                                                print(f'Acc: {acc}')
                                                print(f'Roc Train: {tr_roc1} Validation: {ts_roc1}')

                                            except:
                                                print("Error in this parametrs:")
                                                print(' '.join(['pen', str(pen), 'du', str(du), 't', str(t), 'c', str(c),
                                                                'fit_int', str(fit_int), 'inter', str(inter), 'sol', str(sol),
                                                                'max_it', str(max_it), 'ver', str(ver), 'war', str(war)]))


# logistic regression after find best parameters,
# for calculate AUC of validation and test, and create plot of them
def logistic_regression_test(df_X_tr, df_Y_tr, df_X_ts, df_Y_ts):
    df_x_tr = pd.read_csv(df_X_tr, index_col=0)
    df_y_tr = pd.read_csv(df_Y_tr, index_col=0)
    df_x_ts = pd.read_csv(df_X_ts, index_col=0)
    df_y_ts = pd.read_csv(df_Y_ts, index_col=0)

    df_x_tr, df_y_tr = normalization_zscore(df_x_tr, df_y_tr)
    df_x_ts, df_y_ts = normalization_zscore(df_x_ts, df_y_ts)

    penalty = 'l2'
    dual = False
    tol = 0.0000001
    C = 1.0
    fit_intercept = True
    intercept_scaling = 0.2
    solver = 'newton-cg'
    max_iter = 350
    warm_start = False

    x_train, x_test, y_train, y_test = df_x_tr, df_x_ts, df_y_tr, df_y_ts
    logreg = LogisticRegression(class_weight='balanced', penalty=penalty, dual=dual, tol=tol, C=C,
                                fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, solver=solver,
                                max_iter=max_iter, warm_start=warm_start)
    logreg.fit(x_train, y_train)

    # coefficient of the features
    coef = logreg.coef_
    print("coef:")
    print(coef)

    y_pred = logreg.predict(x_test)
    acc = round(calc_acc(y_test, y_pred), 5)

    y_train_pred = logreg.predict_proba(x_train)[:, 1]
    y_test_pred = logreg.predict_proba(x_test)[:, 1]

    test_results = roc_auc_score(y_test.ravel(), y_test_pred)
    train_results = roc_auc_score(y_train.ravel(), y_train_pred)

    tr_roc1, ts_roc1 = round(train_results, 5), round(test_results, 5)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(' '.join(['pen', str(penalty), 'du', str(dual), 't', str(tol), 'c', str(C),
                    'fit_int', str(fit_intercept), 'inter', str(intercept_scaling), 'sol', str(solver),
                    'max_it', str(max_iter), 'war', str(warm_start)]))

    print(f'Acc: {acc}')
    print(f'Roc Train: {tr_roc1} Test: {ts_roc1}')

    # # Calculate auc for plot
    # base_fpr, base_tpr, _ = roc_curve(y_train.ravel(),
    #                                   [1 for _ in range(len(y_train.ravel()))])
    # val_fpr, val_tpr, _ = roc_curve(y_train.ravel(), y_train_pred1)
    # ts_fpr, ts_tpr, _ = roc_curve(y_test.ravel(), y_pred1)
    #
    # plt.figure(figsize=(8, 6))
    # plt.rcParams['font.size'] = 16
    #
    # # Plot both curves
    # plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    # plt.plot(val_fpr, val_tpr, 'r', label='validation')
    # plt.plot(ts_fpr, ts_tpr, 'g', label='test')
    # plt.legend()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves for LR - 2017')
    # plt.savefig(f'auc_fig_LR/auc_LR_2017.png')
    # plt.show()


# print(" -------------- 2019 ------------------")
# logistic_regression_test("/home/dsi/zuriya/Projects/ML_Donors/split_new/train_x_2019.csv",
#                          "/home/dsi/zuriya/Projects/ML_Donors/split_new/train_y_2019.csv",
#                          "/home/dsi/zuriya/Projects/ML_Donors/split_new/test_x_2019.csv",
#                          "/home/dsi/zuriya/Projects/ML_Donors/split_new/test_y_2019.csv")

logistic_regression_test("/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_x_2019.csv",
                    "/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_y_2019.csv",
                    "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_x_2019.csv",
                    "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_y_2019.csv",)