import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_global_prototypes(model, X_train, y_train, save_path="./data/global_prototypes.pth"):
    model.eval()

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)

    with torch.no_grad():
        embeddings = model(X_train_t)

    all_classes = np.unique(y_train)
    prototypes = []

    for c in all_classes:
        idx = (y_train_t == c)
        proto = embeddings[idx].mean(dim=0)
        prototypes.append(proto)

    prototypes = torch.stack(prototypes)  # shape: [num_classes, embedding_dim]

    torch.save({
        "prototypes": prototypes.cpu(),
        "classes": all_classes
    }, save_path)

    print(f"Global prototypes saved to: {save_path}")

    return prototypes, all_classes
