import torch

def classify_queries(prototypes, query_embeddings):
    distances = euclidean_distance(query_embeddings, prototypes)
    logits = -distances
    return logits


def euclidean_distance(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)
