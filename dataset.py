import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length
    
    def __len__(self):
        return len(self.token_ids) - self.context_length
    
    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.context_length]
        y = self.token_ids[idx + 1 : idx + self.context_length + 1 ]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_dataloader(token_ids, context_length, batch_size, shuffle=True):
    dataset = GPTDataset(token_ids, context_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return dataloader