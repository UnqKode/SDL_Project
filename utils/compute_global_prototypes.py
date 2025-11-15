import torch
import numpy as np
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def compute_global_prototypes(model, X_train, y_train):
    """
    Computes global prototypes using ALL samples of each class.
    """
    model.eval()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    
    with torch.no_grad():
        embeddings = model(X_train_t)

    all_classes = np.unique(y_train)
    n_way = len(all_classes)

    prototypes = []
    
    for class_label in all_classes:
        class_indices = (y_train_t == class_label)
        class_embeds = embeddings[class_indices]
        proto = class_embeds.mean(dim=0)
        prototypes.append(proto)

    prototypes = torch.stack(prototypes)   # shape: [num_classes, embedding_dim]
    torch.save({
        "prototypes": prototypes.cpu(),
        "classes": all_classes
    }, "./data/global_prototypes.pth")

    return prototypes, all_classes
