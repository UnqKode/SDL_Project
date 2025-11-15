from model.arch import AttentionGRUEncoder
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_saved_model(model_path, X_train):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    model = AttentionGRUEncoder(
        input_dim=X_train.shape[1],
        hidden_dim=128,
        num_layers=2,
        embedding_dim=config['embedding_dim'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded successfully!")
    print(f"Test accuracy: {checkpoint['test_accuracy']:.2f}%")
    return model