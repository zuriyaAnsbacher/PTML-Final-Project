import csv
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt


# write results to csv
def write2csv(param, train, validation, output_f):
    out_f = open(output_f, 'a')
    writer = csv.writer(out_f)

    param.extend([train, validation])
    writer.writerow(param)

    out_f.close()


def evaluate_model_test(test_probs, test_y):
    results = {'roc': roc_auc_score(test_y, test_probs)}
    return round(results['roc'], 5)


def evaluate_model(predictions, probs, train_predictions, train_probs, train_labels, test_labels):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    # baseline = {}
    #
    # baseline['recall'] = recall_score(test_labels,
    #                                   [1 for _ in range(len(test_labels))])
    # baseline['precision'] = precision_score(test_labels,
    #                                         [1 for _ in range(len(test_labels))])
    # baseline['roc'] = 0.5

    results = {}

    # results['recall'] = recall_score(test_labels, predictions)
    # results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)

    train_results = {}
    # train_results['recall'] = recall_score(train_labels, train_predictions)
    # train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)

    for metric in ['roc']:
        print(
            f'{metric.capitalize()} Test: {round(results[metric], 5)} Train: {round(train_results[metric], 5)}')

    # # Calculate false positive rates and true positive rates
    # base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    # model_fpr, model_tpr, _ = roc_curve(test_labels, probs)
    #
    # plt.figure(figsize=(8, 6))
    # plt.rcParams['font.size'] = 16
    #
    # # Plot both curves
    # plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    # plt.plot(model_fpr, model_tpr, 'r', label='model')
    # plt.legend()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves - 2017')
    # plt.savefig('auc_fig/auc_xgboost_2017_itr2.png')
    # plt.show()

    return round(train_results['roc'], 5), round(results['roc'], 5)


def xgboost_test(test_x, test_y):
    boost, est, dep, l, g, r, obj = 'gbtree', 160, 12, 0.06, 0.4, 0.8, 'binary:logistic'
    xg_clf = xgb.XGBClassifier(tree_method='exact', booster=boost, max_depth=dep,
                               n_estimators=est, colsample_bytree=0.8, learning_rate=l,
                               gamma=g, reg_lambda=r, objective=obj)

    xg_clf.fit(test_x, test_y.values.ravel())

    # Testing predictions (to determine performance)
    test_probs = xg_clf.predict_proba(test_x)[:, 1]

    # print("------------------------------------")
    # print(', '.join(["boost", str(boost), "est", str(est), "dep", str(dep), "l", str(l),
    #                  "g", str(g), "r", str(r), "obj", str(obj)]))  # + colsample_bytree
    # test_roc = evaluate_model_test(test_probs, test_y)
    #
    # print("boost, est, dep, l, g, r, obj = 'gbtree', 160, 12, 0.06, 0.4, 0.8, 'binary:logistic'")
    # print("roc_test_2019: " + str(test_roc))

    return test_y, test_probs


def xgboost_algorithm(df_x, df_y, output_f):
    # out_f = open(output_f, 'w')
    # writer = csv.writer(out_f)
    #
    # writer.writerow(['booster', 'n_estimators', 'max_depth', 'colsample_bytree', 'learning_rate', 'gamma', "reg_lambda", 'objective',
    #                  'train', 'validation', 'test'])
    #
    # out_f.close()

    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)


    boosterst = ['gbtree']
    n_estimators = [200]  # [160, 210]
    max_depth = [15]  # [15, 18]
    colsample_bytree = [0.8]
    learning_rate = [0.03]  # [0.009, 0.09, 0.3]
    gamma = [0.9]  # [0.5, 0.9]
    reg_lambda = [0.8]  # [0.4, 0.85]
    objective = ['binary:logistic']

    for boost in boosterst:
        for est in n_estimators:
            for dep in max_depth:
                for col in colsample_bytree:
                    for l in learning_rate:
                        for g in gamma:
                            for r in reg_lambda:
                                for obj in objective:
                                    xg_clf = xgb.XGBClassifier(tree_method='exact', booster=boost, max_depth=dep,
                                                               n_estimators=est, colsample_bytree=col, learning_rate=l,
                                                               gamma=g, reg_lambda=r, objective=obj)

                                    xg_clf.fit(X_train, y_train.values.ravel())

                                    feature_importance = xg_clf.feature_importances_
                                    print(feature_importance)
                                    plt.bar(range(len(feature_importance)), feature_importance)
                                    plt.savefig("feature importance fig/feature_importance_2019.png")
                                    plt.show()


                                    # Training predictions (to demonstrate overfitting)
                                    train_rf_predictions = xg_clf.predict(X_train)
                                    train_rf_probs = xg_clf.predict_proba(X_train)[:, 1]

                                    # Testing predictions (to determine performance)
                                    rf_predictions = xg_clf.predict(X_test)
                                    rf_probs = xg_clf.predict_proba(X_test)[:, 1]


                                    print("****************************************")
                                    print(', '.join(["boost", str(boost), "est", str(est), "dep", str(dep), "col",
                                                     str(col), "l", str(l), "g", str(g), "r", str(r),
                                                     "obj", str(obj)]))

                                    train_roc, valid_roc = evaluate_model(rf_predictions, rf_probs, train_rf_predictions,
                                                                          train_rf_probs, y_train, y_test)

                                    return y_test, rf_probs


                                    # if float(valid_roc) > 0.9:
                                    #     write2csv([boost, est, dep, col, l, g, r, obj], train_roc, valid_roc, output_f)
                                    # # plt.savefig('rf_auc_best_parm2.png')


def save_model_to_pickle(df_x, df_y, df_val, pkl_file):
    # create model
    xg_clf = xgb.XGBClassifier(tree_method='exact', booster='gbtree', n_estimators=200,
                               max_depth=15, colsample_bytree=0.8, learning_rate=0.03,
                               gamma=0.9, reg_lambda=0.8, objective='binary:logistic')

    xg_clf.fit(df_x.values, df_y.values)

    # xg_clf.save_model(pkl_file)
    #
    # with open(pkl_file, 'r') as json_f:
    #     xg_clf_loaded = json.load(json_f)

    pickle.dump(xg_clf, open(pkl_file, "wb"))

    # xg_clf_loaded = pickle.load(open(pick_file, "rb"))
    #
    # # test
    # test_orig = xg_clf.predict_proba(df_val)[:, 1]
    # test_load = xg_clf_loaded.predict_proba(df_val)[:, 1]


def auc_validation_test(df_x_tr, df_y_tr, df_x_ts, df_y_ts):


    # split train to train and validation
    x_train, x_val, y_train, y_val = train_test_split(df_x_tr, df_y_tr, test_size=0.2)

    # x_train, x_val, y_train, y_val = df_x_tr, df_x_ts, df_y_tr, df_y_ts

    # # create model
    # xg_clf = xgb.XGBClassifier(tree_method='exact', booster='gbtree', n_estimators=200,
    #                            max_depth=15, colsample_bytree=0.8, learning_rate=0.03,
    #                            gamma=0.9, reg_lambda=0.8, objective='binary:logistic')

    # create model
    xg_clf = xgb.XGBClassifier(tree_method='exact', booster='gbtree', n_estimators=200,
                               max_depth=15, colsample_bytree=0.8, learning_rate=0.03,
                               gamma=0.9, reg_lambda=0.8, objective='binary:logistic')

    # fit model
    xg_clf.fit(x_train.values, y_train.values.ravel())

    # tmp
    pickle.dump(xg_clf, open("save model/xgboost_model_2019_2.pkl", "wb"))

    """
    feature_importance = xg_clf.feature_importances_
    print("feature_importance")
    print(feature_importance)
    """

    tr_y_probs = xg_clf.predict_proba(x_train)[:, 1]
    val_y_probs = xg_clf.predict_proba(x_val)[:, 1]     # prob validation
    tst_y_probs = xg_clf.predict_proba(df_x_ts)[:, 1]     # prob test

    # calculate auc score and print
    res_auc_tr = roc_auc_score(y_train, tr_y_probs)
    res_auc_val = roc_auc_score(y_val, val_y_probs)
    res_auc_ts = roc_auc_score(df_y_ts, tst_y_probs)

    print(f'Train: {round(res_auc_tr, 5)}, Validation: {round(res_auc_val, 5)} , Test: {round(res_auc_ts, 5)}')
    # print(f'Train: {round(res_auc_tr, 5)}, Test: {round(res_auc_val, 5)}')


    # # Calculate auc for plot
    # base_fpr, base_tpr, _ = roc_curve(y_val, [1 for _ in range(len(y_val))])
    # tr_fpr, tr_tpr, _ = roc_curve(y_train, tr_y_probs)
    # val_fpr, val_tpr, _ = roc_curve(y_val, val_y_probs)
    # ts_fpr, ts_tpr, _ = roc_curve(df_y_ts, tst_y_probs)
    #
    # plt.figure(figsize=(8, 6))
    # plt.rcParams['font.size'] = 16
    #
    # # Plot both curves
    # plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    # plt.plot(tr_fpr, tr_tpr, 'm', label='train')
    # plt.plot(val_fpr, val_tpr, 'r', label='validation')
    # plt.plot(ts_fpr, ts_tpr, 'g', label='test')
    # plt.legend()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves of XGBoost - 2016')
    # plt.savefig('auc_xgboost_new/auc_xgboost_2016_with_train.png')
    # plt.show()
