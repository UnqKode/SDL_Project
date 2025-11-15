import torch
import torch.optim as optim
import torch.nn as nn
from utils.compute_prototypes import compute_prototypes
from utils.classify_queries import classify_queries
from tqdm import tqdm

class ProtoNetTrainer:
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-4, scheduler_step=10, scheduler_gamma=0.5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_episode(self, support_data, support_labels, query_data, query_labels):
        self.model.train()
        support_data = support_data.to(self.device)
        support_labels = support_labels.to(self.device)
        query_data = query_data.to(self.device)
        query_labels = query_labels.to(self.device)
        support_embeddings = self.model(support_data)
        query_embeddings = self.model(query_data)
        n_way = torch.unique(support_labels).size(0)
        prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
        logits = classify_queries(prototypes, query_embeddings)
        loss = self.criterion(logits, query_labels)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == query_labels).float().mean().item()
        return loss.item(), accuracy

    def train_epoch(self, train_loader, epoch):
        total_loss = 0
        total_acc = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for episode_data in pbar:
            support_data, support_labels, query_data, query_labels = episode_data
            support_data = support_data.squeeze(0)
            support_labels = support_labels.squeeze(0)
            query_data = query_data.squeeze(0)
            query_labels = query_labels.squeeze(0)
            loss, acc = self.train_episode(support_data, support_labels, query_data, query_labels)
            total_loss += loss
            total_acc += acc
            pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc*100:.2f}%'})
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)
        self.scheduler.step()
        return avg_loss, avg_acc

    def evaluate(self, val_loader):
        self.model.eval()
        total_acc = 0
        with torch.no_grad():
            for episode_data in tqdm(val_loader, desc='Validation'):
                support_data, support_labels, query_data, query_labels = episode_data
                support_data = support_data.squeeze(0).to(self.device)
                support_labels = support_labels.squeeze(0).to(self.device)
                query_data = query_data.squeeze(0).to(self.device)
                query_labels = query_labels.squeeze(0).to(self.device)
                support_embeddings = self.model(support_data)
                query_embeddings = self.model(query_data)
                n_way = torch.unique(support_labels).size(0)
                prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
                logits = classify_queries(prototypes, query_embeddings)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == query_labels).float().mean().item()
                total_acc += acc
        avg_acc = total_acc / len(val_loader)
        self.val_accs.append(avg_acc)
        return avg_acc

    def fit(self, train_loader, val_loader, n_epochs):
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        best_val_acc = 0
        best_model_state = None
        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_acc = self.evaluate(val_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f" New best validation accuracy: {val_acc*100:.2f}%")
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Acc:  {train_acc*100:.2f}%")
            print(f"  Val Acc:    {val_acc*100:.2f}%")
            print(f"  LR:         {self.scheduler.get_last_lr()[0]:.6f}")
            print("-" * 60)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n Loaded best model (val_acc: {best_val_acc*100:.2f}%)")
        return best_val_acc
