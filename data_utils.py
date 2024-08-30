#%%
import pickle
from torch_geometric.data import Dataset, Batch
import numpy as np


class GraphSetDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)
        
        targets = np.array([self.data_list[i][-1] for i in range(len(self.data_list))])
        self.target_mean = np.mean(targets)
        self.target_std = np.std(targets)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        sample = self.data_list[idx]
        normalized_target = [float((sample[-1] - self.target_mean) / self.target_std)]
        normalized_sample = sample[:-1] + (normalized_target,)
        return normalized_sample

    def get_orig(self, target):
        return target * self.target_std + self.target_mean


def graph_set_collate(batch):
    graph_lists, ys = zip(*batch)
    batched_graph_sets = [Batch.from_data_list(g_list) for g_list in graph_lists]
    return batched_graph_sets, ys
