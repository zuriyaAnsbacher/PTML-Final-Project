import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

from train_test_data import trainData, testData
from binary_classification import binaryClassification1, binaryClassification2, binaryClassification3, binaryClassification4


def normalization_zscore(df_x, df_y):
    df_x_without_norm = df_x['donor_mass_log']
    df_x.drop(['donor_mass_log'], axis=1, inplace=True)

    df_x_without_norm, df_x, df_y = df_x_without_norm.values, df_x.values, df_y.values

    df_x = zscore(df_x)
    columns = [df_x[:, i] for i in range(len(df_x[0]))]  # split to columns list
    columns.append(df_x_without_norm)  # add the column without normalization (donor_mass_log)
    columns = [col.reshape(len(col), 1) for col in columns]  # (len, ) to (len, 1)
    return np.concatenate(columns, axis=1), df_y


def evaluate_model(train_probs, train_labels, test_probs, test_labels):
    train_roc = roc_auc_score(train_labels, train_probs)
    # test_roc = roc_auc_score(test_labels, test_probs)  # test_probs[:1]

    # # Calculate auc for plot
    # base_fpr, base_tpr, _ = roc_curve(train_labels, [1 for _ in range(len(train_labels))])
    # val_fpr, val_tpr, _ = roc_curve(train_labels, train_probs)
    # ts_fpr, ts_tpr, _ = roc_curve(test_labels, test_probs)
    #
    # # plt.figure(figsize=(8, 6))
    # # plt.rcParams['font.size'] = 16
    # #
    # # # Plot both curves
    # # plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    # # plt.plot(val_fpr, val_tpr, 'r', label='validation')
    # # plt.plot(ts_fpr, ts_tpr, 'g', label='test')
    # # plt.legend()
    # # plt.xlabel('False Positive Rate')
    # # plt.ylabel('True Positive Rate')
    # # plt.title('ROC Curves for NN - 2016')
    # # plt.savefig('auc_fig_NN/auc_NN_2016.png')
    # # plt.show()


    # return round(train_roc, 5), round(test_roc, 5)
    return round(train_roc, 5)


def binary_acc(y_pred, y_test):
    y_pred_tag = (y_pred > 0.5).float()

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


def main_NN_with_kfold(df_X, df_Y):

    EPOCH_LST = [10]  # 40 epoch-> Roc Test: 0.87796 , 70-> Roc Test: 0.87258, 15-> 0.87916
    BATCH_SIZE = 128
    LEARNING_RATE_LST = [0.00001]

    df_x = pd.read_csv(df_X, index_col=0)
    df_y = pd.read_csv(df_Y, index_col=0)

    df = pd.concat([df_x, df_y], axis=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=2)

    for ind_model, binaryClassification in enumerate([binaryClassification2, binaryClassification3, binaryClassification4]):
        for EPOCH in EPOCH_LST:
            for LEARNING_RATE in LEARNING_RATE_LST:

                acc_lst, loss_lst, auc_train, auc_test = [], [], [], []
                for train, test in kf.split(df):

                    df_train = df.iloc[train]
                    df_test = df.iloc[test]

                    y_train = df_train['donated'].values
                    x_train = df_train.drop(['donated'], axis=1).values

                    y_test = df_test['donated'].values
                    x_test = df_test.drop(['donated'], axis=1).values

                    # df_x = df_x.values
                    # df_y = df_y.values
                    # # df_x = zscore(df_x)  # does it make it on each column?
                    #
                    # x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)  # random_state= 69?

                    # normalization
                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(x_train)
                    x_test = scaler.fit_transform(x_test)

                    train_data = trainData(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
                    test_data = testData(torch.FloatTensor(x_test))

                    train_data_to_auc = testData(torch.FloatTensor(x_train))

                    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
                    test_loader = DataLoader(dataset=test_data, batch_size=1)

                    train_loader_to_auc = DataLoader(dataset=train_data_to_auc, batch_size=1)

                    device = 'cpu'
                    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                    print("device: " + str(device))

                    model = binaryClassification()
                    model.to(device)

                    print(model)

                    criterion = nn.BCELoss()
                    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

                    model.train()

                    for e in range(1, EPOCH + 1):
                        epoch_loss = 0
                        epoch_acc = 0

                        for x_batch, y_batch in train_loader:
                            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                            # y_batch = y_batch.long()
                            optimizer.zero_grad()

                            y_pred = model(x_batch)

                            loss = criterion(y_pred, y_batch.unsqueeze(1))
                            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

                            loss.backward()
                            optimizer.step()

                            epoch_loss += loss.item()
                            epoch_acc += acc.item()

                        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
                    acc_lst.append(epoch_acc/len(train_loader))
                    loss_lst.append(epoch_loss/len(train_loader))

                    y_pred_train_list = []
                    y_probability = []

                    model.eval()
                    with torch.no_grad():
                        for x_batch in test_loader:
                            x_batch = x_batch.to(device)
                            y_test_pred = model(x_batch)

                            y_probability.append(y_test_pred.cpu().numpy())

                            y_pred_tag = torch.round(y_test_pred)

                        # I added
                        for x_batch_train in train_loader_to_auc:
                            x_batch_train = x_batch_train.to(device)
                            y_train_pred = model(x_batch_train)
                            y_pred_train_list.append(y_train_pred.cpu().numpy())

                    y_probability = np.array([a.squeeze().tolist() for a in y_probability])
                    # # I added
                    y_pred_train_list = np.array([b.squeeze().tolist() for b in y_pred_train_list])

                    # eval by auc
                    # # I added
                    # tr_roc, ts_roc = evaluate_model(y_pred_train_list, y_train, y_pred_list, y_test)

                    tr_roc, ts_roc = evaluate_model(y_pred_train_list, y_train.ravel(), y_probability, y_test.ravel())
                    auc_train.append(tr_roc)
                    auc_test.append(ts_roc)

                print(f'--------------- Model: {ind_model + 2} Epoch num: {EPOCH} Learning Rate: {LEARNING_RATE} ---------------')
                print("avg_acc: ", str(sum(acc_lst) / len(acc_lst)))
                print("avg_loss: ", str(sum(loss_lst) / len(loss_lst)))
                print("avg_auc_train: ", str(sum(auc_train) / len(auc_train)))
                print("avg_auc_validation: ", str(sum(auc_test) / len(auc_test)))


def main_NN_without_kfold(df_X, df_Y):
    EPOCH_LST = [10, 15]  # 40 epoch-> Roc Test: 0.87796 , 70-> Roc Test: 0.87258, 15-> 0.87916
    BATCH_SIZE = 128
    LEARNING_RATE_LST = [0.00001, 0.0001]

    df_x = pd.read_csv(df_X, index_col=0)
    df_y = pd.read_csv(df_Y, index_col=0)

    df_x = df_x.values
    df_y = df_y.values
    df_x = zscore(df_x)  # does it make it on each column?

    for ind_model, binaryClassification in enumerate([binaryClassification1, binaryClassification2]):
        for EPOCH in EPOCH_LST:
            for LEARNING_RATE in LEARNING_RATE_LST:

                x_train, x_validation, y_train, y_validation = train_test_split(df_x, df_y, test_size=0.3)  # random_state= 69?

                # # normalization
                # scaler = StandardScaler()
                # x_train = scaler.fit_transform(x_train)
                # x_validation = scaler.fit_transform(x_validation)

                train_data = trainData(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
                validation_data = testData(torch.FloatTensor(x_validation))

                train_data_to_auc = testData(torch.FloatTensor(x_train))

                train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
                validation_loader = DataLoader(dataset=validation_data, batch_size=1)

                train_loader_to_auc = DataLoader(dataset=train_data_to_auc, batch_size=1)

                device = 'cpu'
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                print("device: " + str(device))

                model = binaryClassification()
                model.to(device)

                print(model)

                # criterion = nn.CrossEntropyLoss(reduction='mean')
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

                model.train()

                for e in range(1, EPOCH + 1):
                    epoch_loss = 0
                    epoch_acc = 0

                    for x_batch, y_batch in train_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        # y_batch = y_batch.long()
                        optimizer.zero_grad()

                        y_pred = model(x_batch)

                        # # loss = criterion(y_pred, torch.max(y_batch, 1)[1])  # in orig: y_batch.unsqueeze(1)
                        # loss = criterion(y_pred, y_batch)  # in orig: y_batch.unsqueeze(1)
                        # acc = binary_acc(y_pred, y_batch)  # in orig: y_batch.unsqueeze(1)

                        loss = criterion(y_pred, y_batch)
                        acc = binary_acc(y_pred, y_batch)

                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        epoch_acc += acc.item()

                    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
                # acc_lst.append(epoch_acc/len(train_loader))
                # loss_lst.append(epoch_loss/len(train_loader))

                y_pred_train_list = []
                y_pred_validation_list = []

                model.eval()
                with torch.no_grad():
                    for x_batch in validation_loader:
                        x_batch = x_batch.to(device)
                        y_validation_pred = model(x_batch)
                        y_pred_validation_list.append(y_validation_pred.cpu().numpy())

                    # # I added - for evaluate train
                    # for x_batch_train in train_loader_to_auc:
                    #     x_batch_train = x_batch_train.to(device)
                    #     y_train_pred = model(x_batch_train)
                    #     y_pred_train_list.append(y_train_pred.cpu().numpy())

                y_pred_validation_list = np.array([a.squeeze().tolist() for a in y_pred_validation_list])

                # # I added
                y_pred_train_list = np.array([b.squeeze().tolist() for b in y_pred_train_list])

                # eval by auc
                tr_roc, ts_roc = evaluate_model(y_pred_train_list, y_train.ravel(), y_pred_validation_list, y_validation.ravel())
                # auc_train.append(tr_roc)
                # auc_test.append(ts_roc)

                print(f'--------------- Model: {ind_model + 1} Epoch num: {EPOCH} Learning Rate: {LEARNING_RATE} ---------------')
                print(f'AUC Train: {tr_roc} AUC Validation: {ts_roc}')


def main_NN_with_test(df_X_tr, df_Y_tr, df_X_ts, df_Y_ts):

    EPOCH_LST = [10]
    BATCH_SIZE = 128
    LEARNING_RATE_LST = [0.0001]

    df_x_tr = pd.read_csv(df_X_tr, index_col=0)
    df_y_tr = pd.read_csv(df_Y_tr, index_col=0)
    df_x_ts = pd.read_csv(df_X_ts, index_col=0)
    df_y_ts = pd.read_csv(df_Y_ts, index_col=0)

    df_x_tr, df_y_tr = normalization_zscore(df_x_tr, df_y_tr)
    df_x_ts, df_y_ts = normalization_zscore(df_x_ts, df_y_ts)

    for ind_model, binaryClassification in enumerate([binaryClassification2]):
        for EPOCH in EPOCH_LST:
            for LEARNING_RATE in LEARNING_RATE_LST:

                x_train, x_validation, y_train, y_validation = train_test_split(df_x_tr, df_y_tr, test_size=0.3)  # random_state= 69?
                x_test, y_test = df_x_ts, df_y_ts

                # # normalization
                # scaler = StandardScaler()
                # x_train = scaler.fit_transform(x_train)
                # x_validation = scaler.fit_transform(x_validation)

                train_data = trainData(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
                validation_data = testData(torch.FloatTensor(x_validation))
                test_data = testData(torch.FloatTensor(x_test))

                train_data_to_auc = testData(torch.FloatTensor(x_train))

                train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
                validation_loader = DataLoader(dataset=validation_data, batch_size=1)
                test_loader = DataLoader(dataset=test_data, batch_size=1)

                train_loader_to_auc = DataLoader(dataset=train_data_to_auc, batch_size=1)

                device = 'cpu'
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                print("device: " + str(device))

                model = binaryClassification()
                model.to(device)

                print(model)

                # criterion = nn.CrossEntropyLoss(reduction='mean')
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

                model.train()

                for e in range(1, EPOCH + 1):
                    epoch_loss = 0
                    epoch_acc = 0

                    for x_batch, y_batch in train_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        # y_batch = y_batch.long()
                        optimizer.zero_grad()

                        y_pred = model(x_batch)

                        loss = criterion(y_pred, y_batch)
                        acc = binary_acc(y_pred, y_batch)

                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        epoch_acc += acc.item()

                    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

                y_pred_train_list = []
                y_pred_validation_list = []
                y_pred_test_list = []

                model.eval()
                with torch.no_grad():
                    for x_batch_train in train_loader_to_auc:
                        x_batch_train = x_batch_train.to(device)
                        y_train_pred = model(x_batch_train)
                        y_pred_train_list.append(y_train_pred.cpu().numpy())

                    for x_batch in validation_loader:
                        x_batch = x_batch.to(device)
                        y_validation_pred = model(x_batch)
                        y_pred_validation_list.append(y_validation_pred.cpu().numpy())

                    for x_batch_test in test_loader:
                        x_batch_test = x_batch_test.to(device)
                        y_test_pred = model(x_batch_test)
                        y_pred_test_list.append(y_test_pred.cpu().numpy())

                y_pred_train_list = np.array([b.squeeze().tolist() for b in y_pred_train_list])
                y_pred_validation_list = np.array([a.squeeze().tolist() for a in y_pred_validation_list])
                y_pred_test_list = np.array([c.squeeze().tolist() for c in y_pred_test_list])

                # eval by auc
                # val_roc, ts_roc = evaluate_model(y_pred_validation_list, y_validation.ravel(), y_pred_test_list, y_test.ravel())
                tr_roc = evaluate_model(y_pred_train_list, y_train.ravel(), None, None)
                val_roc = evaluate_model(y_pred_validation_list, y_validation.ravel(), None, None)
                ts_roc = evaluate_model(y_pred_test_list, y_test.ravel(), None, None)


                print(f'--------------- Model: {ind_model + 1} Epoch num: {EPOCH} Learning Rate: {LEARNING_RATE} ---------------')
                print(f'AUC Train: {tr_roc}, Validation: {val_roc}, Test: {ts_roc}')


def mini_data(tr_x_file, tr_y_file, rows2remove, out_x, out_y):

    df_x = pd.read_csv(tr_x_file, index_col=0)
    df_y = pd.read_csv(tr_y_file, index_col=0)

    df = pd.concat([df_x, df_y], axis=1)

    ind_class0 = np.where(df['donated'] == 0)
    ind2remove = random.sample(set(ind_class0[0].flatten()), rows2remove)

    rows = df.index[[x for x in ind2remove]]

    df_new = df.drop(rows)

    # check:
    print('check:')
    print(df_new['donated'].value_counts())

    df_y_new = pd.DataFrame(df_new['donated'])
    df_new.drop(['donated'], axis=1, inplace=True)

    df_new.to_csv(out_x)
    df_y_new.to_csv(out_y)


# print("********************* 2016 *************************")
# main_NN("data/train_x_2016.csv",  "data/train_y_2016.csv")

# mini_data("/home/dsi/zuriya/Projects/ML_Donor_2016/split_new/test_x_2016.csv", "/home/dsi/zuriya/Projects/ML_Donor_2016/split_new/test_y_2016.csv", 1400000, "mini_data/mini_testX2016.csv",
#           "mini_data/mini_testY2016.csv")


# print("********** 2016 *************")
# main_NN_with_test("/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_x_2016.csv",
#                   "/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_y_2016.csv",
#                   "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_x_2016.csv",
#                   "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_y_2016.csv")
#
# print("********** 2017 *************")
# main_NN_with_test("/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_x_2017.csv",
#                   "/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_y_2017.csv",
#                   "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_x_2017.csv",
#                   "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_y_2017.csv")
#
# print("********** 2018 *************")
# main_NN_with_test("/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_x_2018.csv",
#                   "/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_y_2018.csv",
#                   "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_x_2018.csv",
#                   "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_y_2018.csv")

print("********** 2019 *************")
main_NN_with_test("/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_x_2019.csv",
                  "/home/dsi/zuriya/Projects/ML_Donors/split_new1/train_y_2019.csv",
                  "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_x_2019.csv",
                  "/home/dsi/zuriya/Projects/ML_Donors/split_new1/test_y_2019.csv")