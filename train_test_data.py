from torch.utils.data import Dataset, DataLoader


class trainData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


class testData(Dataset):
    def __init__(self, x_data):
        self.x_data = x_data

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return len(self.x_data)

