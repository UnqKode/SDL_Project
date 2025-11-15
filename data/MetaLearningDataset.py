import torch
import random
from torch.utils.data import Dataset

class MetaLearningDataset(Dataset):
    def __init__(self, X, y, n_way, k_shot, n_query, n_episodes):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.class_to_indices = {}
        for idx, label in enumerate(y):
            label = int(label)
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())
        for cls in self.classes:
            if len(self.class_to_indices[cls]) < (k_shot + n_query):
                raise ValueError(
                    f"Class {cls} has only {len(self.class_to_indices[cls])} samples, need at least {k_shot + n_query}"
                )

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        selected_classes = random.sample(self.classes, self.n_way)
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        label_map = {cls: i for i, cls in enumerate(selected_classes)}
        for cls in selected_classes:
            class_indices = self.class_to_indices[cls]
            selected_indices = random.sample(class_indices, self.k_shot + self.n_query)
            support_idx = selected_indices[:self.k_shot]
            query_idx = selected_indices[self.k_shot:]
            for idx in support_idx:
                support_data.append(self.X[idx])
                support_labels.append(label_map[cls])
            for idx in query_idx:
                query_data.append(self.X[idx])
                query_labels.append(label_map[cls])
        return (
            torch.stack(support_data),
            torch.LongTensor(support_labels),
            torch.stack(query_data),
            torch.LongTensor(query_labels)
        )