import numpy as np
import random
import torch
from tqdm import tqdm
from utils.compute_prototypes import compute_prototypes
from utils.classify_queries import classify_queries
import torch.nn.functional as F

def test_model(model, X_test, y_test, X_train, y_train, k_shot=5, device='cuda'):
    model.eval()
    all_classes = np.unique(y_train)
    n_way = len(all_classes)
    print("\n" + "="*60)
    print(f"TESTING: {n_way}-way {k_shot}-shot classification")
    print("="*60)
    train_class_to_indices = {}
    for idx, label in enumerate(y_train):
        if label not in train_class_to_indices:
            train_class_to_indices[label] = []
        train_class_to_indices[label].append(idx)
    support_data_list = []
    support_labels_list = []
    for class_idx, class_label in enumerate(all_classes):
        class_indices = train_class_to_indices[class_label]
        if len(class_indices) < k_shot:
            selected_indices = class_indices
        else:
            selected_indices = random.sample(class_indices, k_shot)
        for idx in selected_indices:
            support_data_list.append(X_train[idx])
            support_labels_list.append(class_idx)
    support_data = torch.FloatTensor(np.array(support_data_list)).to(device)
    support_labels = torch.LongTensor(support_labels_list).to(device)
    label_to_idx = {label: idx for idx, label in enumerate(all_classes)}
    mapped_test_labels = np.array([label_to_idx[label] for label in y_test])
    test_data = torch.FloatTensor(X_test).to(device)
    test_labels = torch.LongTensor(mapped_test_labels).to(device)
    with torch.no_grad():
        support_embeddings = model(support_data)
        prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
        batch_size = 128
        all_preds = []
        all_probs = []
        for i in tqdm(range(0, len(test_data), batch_size), desc='Testing'):
            batch = test_data[i:i+batch_size]
            batch_embeddings = model(batch)
            logits = classify_queries(prototypes, batch_embeddings)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)
        correct = (all_preds == test_labels.cpu()).sum().item()
        total = len(test_labels)
        accuracy = (correct / total) * 100
        class_accs = []
        print("\n" + "-"*60)
        print("Per-Class Results:")
        print("-"*60)
        activity_labels = {
            0: 'WALKING', 1: 'WALKING_UPSTAIRS',
            2: 'WALKING_DOWNSTAIRS', 3: 'SITTING',
            4: 'STANDING', 5: 'LAYING'
        }
        for idx, class_label in enumerate(all_classes):
            class_mask = (test_labels.cpu() == idx)
            if class_mask.sum() > 0:
                class_correct = (all_preds[class_mask] == idx).sum().item()
                class_total = class_mask.sum().item()
                class_acc = (class_correct / class_total) * 100
                class_accs.append(class_acc)
                print(f"{activity_labels[class_label]:20s}: {class_acc:6.2f}% ({class_correct:4d}/{class_total:4d})")
        print("-"*60)
        print(f"Overall Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Mean Class Accuracy:   {np.mean(class_accs):.2f}%")
        print("="*60)
    return accuracy, all_preds.numpy(), all_probs.numpy()