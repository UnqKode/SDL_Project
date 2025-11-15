import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.compute_prototypes import compute_prototypes
from utils.classify_queries import classify_queries
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_single_sample(model, sample, X_train, y_train, k_shot=5):
    model.eval()
    all_classes = np.unique(y_train)
    n_way = len(all_classes)
    train_class_to_indices = {}
    for idx, label in enumerate(y_train):
        if label not in train_class_to_indices:
            train_class_to_indices[label] = []
        train_class_to_indices[label].append(idx)
    support_data_list = []
    support_labels_list = []
    for class_idx, class_label in enumerate(all_classes):
        class_indices = train_class_to_indices[class_label]
        selected_indices = random.sample(class_indices, min(k_shot, len(class_indices)))
        for idx in selected_indices:
            support_data_list.append(X_train[idx])
            support_labels_list.append(class_idx)
    support_data = torch.FloatTensor(np.array(support_data_list)).to(device)
    support_labels = torch.LongTensor(support_labels_list).to(device)
    query_data = torch.FloatTensor(sample).unsqueeze(0).to(device)
    with torch.no_grad():
        support_embeddings = model(support_data)
        query_embedding = model(query_data)
        prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
        logits = classify_queries(prototypes, query_embedding)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_class].item()
    original_label = all_classes[pred_class]
    activity_labels = {
        0: 'WALKING', 1: 'WALKING_UPSTAIRS',
        2: 'WALKING_DOWNSTAIRS', 3: 'SITTING',
        4: 'STANDING', 5: 'LAYING'
    }
    return original_label, activity_labels[original_label], confidence
