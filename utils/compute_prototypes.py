import torch

def compute_prototypes(support_embeddings, support_labels, n_way):
    embedding_dim = support_embeddings.size(-1)
    prototypes = torch.zeros(n_way, embedding_dim, device=support_embeddings.device)
    for c in range(n_way):
        class_mask = (support_labels == c)
        class_embeddings = support_embeddings[class_mask]
        if class_embeddings.size(0) > 0:
            prototypes[c] = class_embeddings.mean(dim=0)
        else:
            print(f"Warning: No samples for class {c}")
    return prototypes
