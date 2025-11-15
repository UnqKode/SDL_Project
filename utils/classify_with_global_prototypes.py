import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def classify_with_global_prototypes(model, prototypes, all_classes, X_test, y_test):
    model.eval()

    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    correct = 0
    total = len(X_test)

    with torch.no_grad():
        test_embeddings = model(X_test_t)

        # Compute distances: [num_test, num_classes]
        dists = torch.cdist(test_embeddings, prototypes)

        preds = torch.argmin(dists, dim=1)

    y_pred_labels = all_classes[preds.cpu().numpy()]
    accuracy = (y_pred_labels == y_test).mean() * 100

    return accuracy, y_pred_labels
