import os
import torch
from torch.utils.data import DataLoader
from data.dataloader import DataLoader_UCIHAR
from data.MetaLearningDataset import MetaLearningDataset
from utils.plot_confusion_matrix import plot_confusion_matrix
from utils.test_model import test_model
from utils.plot_training_history import plot_training_history
from model.arch import AttentionGRUEncoder
from model.protonetTrainer import ProtoNetTrainer
from config import CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "./data/UCI-HAR Dataset"

def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(" UCI-HAR dataset not found")
    data_loader = DataLoader_UCIHAR(DATA_PATH)
    X_train, y_train, X_test, y_test = data_loader.load_all_data()
    data_loader.get_data_statistics(X_train, y_train, X_test, y_test)
    for key, value in CONFIG.items():
        print(f"{key:20s}: {value}")
    
    train_dataset = MetaLearningDataset(
        X_train, y_train,
        n_way=CONFIG['n_way'],
        k_shot=CONFIG['k_shot'],
        n_query=CONFIG['n_query'],
        n_episodes=CONFIG['n_train_episodes']
    )

    val_dataset = MetaLearningDataset(
        X_train, y_train,
        n_way=CONFIG['n_way'],
        k_shot=CONFIG['k_shot'],
        n_query=CONFIG['n_query'],
        n_episodes=CONFIG['n_val_episodes']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = AttentionGRUEncoder(
        input_dim=X_train.shape[2],
        hidden_dim=128,
        num_layers=2,
        embedding_dim=CONFIG['embedding_dim'],
        dropout=CONFIG['dropout']
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model created with {n_params:,} trainable parameters")
    print(f" Model architecture: {model.__class__.__name__}")

    trainer = ProtoNetTrainer(
        model=model,
        device=device,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        scheduler_step=CONFIG['scheduler_step'],
        scheduler_gamma=CONFIG['scheduler_gamma']
    )

    best_val_acc = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=CONFIG['n_epochs']
    )

    print(f"\n Training completed!")
    print(f" Best validation accuracy: {best_val_acc*100:.2f}%")

    print("\nPlotting training history...")
    plot_training_history(trainer)

    print("\nTesting model on test set...")
    test_accuracy, test_preds, test_probs = test_model(
        model=trainer.model,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        k_shot=CONFIG['test_k_shot'],
        device=device # type: ignore
    )

    print("\nGenerating visualizations...")
    activity_labels = {
        0: 'WALKING', 1: 'WALKING_UPSTAIRS',
        2: 'WALKING_DOWNSTAIRS', 3: 'SITTING',
        4: 'STANDING', 5: 'LAYING'
    }
    plot_confusion_matrix(y_test, test_preds, activity_labels)

    print("\nSaving model...")
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': CONFIG,
        'test_accuracy': test_accuracy,
        'best_val_acc': best_val_acc
    }, './model/protonet_har_model.pth')
    print(" Model saved as 'protonet_har_model.pth'")

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Model Architecture:      {model.__class__.__name__}")
    print(f"Total Parameters:        {n_params:,}")
    print(f"Training Episodes:       {CONFIG['n_train_episodes']}")
    print(f"Validation Episodes:     {CONFIG['n_val_episodes']}")
    print(f"Meta-learning Setup:     {CONFIG['n_way']}-way {CONFIG['k_shot']}-shot")
    print(f"Best Validation Acc:     {best_val_acc*100:.2f}%")
    print(f"Test Accuracy:           {test_accuracy:.2f}%")
    print("="*60)

    return trainer.model, test_accuracy
