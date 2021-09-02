import torch
import torch.nn as nn
import torch.nn.functional as F


class binaryClassification1(nn.Module):
    def __init__(self):
        super(binaryClassification1, self).__init__()

        self.layer_1 = nn.Linear(8, 200)
        self.layer_2 = nn.Linear(200, 50)
        self.layer_out = nn.Linear(50, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.batchnorm2 = nn.BatchNorm1d(50)

    def forward(self, inputs):

        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)  # is it do something?
        x = self.layer_out(x)
        x = torch.sigmoid(x)

        return x

# change in hidden layers size and activation function
class binaryClassification2(nn.Module):
    def __init__(self):
        super(binaryClassification2, self).__init__()

        self.layer_1 = nn.Linear(8, 500)
        self.layer_2 = nn.Linear(500, 100)
        self.layer_out = nn.Linear(100, 1)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(500)
        self.batchnorm2 = nn.BatchNorm1d(100)

    def forward(self, inputs):

        x = self.tanh(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.tanh(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)  # is it do something?
        x = self.layer_out(x)
        x = torch.sigmoid(x)

        return x

# change in activation function and dropout size
class binaryClassification3(nn.Module):
    def __init__(self):
        super(binaryClassification3, self).__init__()

        self.layer_1 = nn.Linear(8, 200)
        self.layer_2 = nn.Linear(200, 50)
        self.layer_out = nn.Linear(50, 1)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.batchnorm2 = nn.BatchNorm1d(50)

    def forward(self, inputs):

        x = self.tanh(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.tanh(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)  # is it do something?
        x = self.layer_out(x)
        x = torch.sigmoid(x)

        return x


# no dropout, no batchNorm in layers 2
class binaryClassification4(nn.Module):
    def __init__(self):
        super(binaryClassification4, self).__init__()

        self.layer_1 = nn.Linear(8, 200)
        self.layer_2 = nn.Linear(200, 50)
        self.layer_out = nn.Linear(50, 1)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(200)

    def forward(self, inputs):

        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        x = torch.sigmoid(x)

        return x
