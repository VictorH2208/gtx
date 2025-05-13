import torch
from torch.utils.data import Dataset

class FluorescenceDataset(Dataset):

    def __init__(self, data):
        """
        fluorescence: (N, H, W, F)
        mu_a, mu_s: (N, H, W)
        concentration_fluor: (N, H, W)
        depth: (N, H, W)
        """
        super().__init__()
        self.fluorescence = torch.tensor(data['fluorescence'], dtype=torch.float32) 
        self.mu_a = torch.tensor(data['mu_a'], dtype=torch.float32)  
        self.mu_s = torch.tensor(data['mu_s'], dtype=torch.float32)
        self.concentration = torch.tensor(data['concentration_fluor'], dtype=torch.float32)
        self.depth = torch.tensor(data['depth'], dtype=torch.float32)

    def __len__(self):
        return self.fluorescence.shape[0]

    def __getitem__(self, idx):
        return self.fluorescence[idx], self.mu_a[idx], self.mu_s[idx], self.concentration[idx], self.depth[idx]
