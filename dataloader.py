# dataloader.py
from torch.utils.data import Dataset

class IrisDataset(Dataset):
    def __init__(self, X_dict_list, y_series):
        """
        X_dict_list: list of dicts (e.g., [{"feature1": val1, "feature2": val2, ...}, ...])
        y_series: pandas Series of string labels (e.g., "setosa", "virginica")
        """
        self.X = X_dict_list
        self.y = y_series.reset_index(drop=True)  # 避免 index 錯位
        #self.y = y_series

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return {
            "input": self.X[idx],     # dict (for textgrad)
            "target": self.y[idx]  # string label
        }
