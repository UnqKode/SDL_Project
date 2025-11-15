import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGRUEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, embedding_dim=128, dropout=0.3):
        super(AttentionGRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        gru_output_dim = hidden_dim * 2
        self.attention = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc1 = nn.Linear(gru_output_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attention_weights = self.attention(gru_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(gru_out * attention_weights, dim=1)
        out = F.relu(self.bn1(self.fc1(context)))
        out = self.dropout1(out)
        embedding = self.bn2(self.fc2(out))
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding