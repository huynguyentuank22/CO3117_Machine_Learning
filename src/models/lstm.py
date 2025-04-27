import torch
import torch.nn as nn
# from cfg import CFGs

class SentimentLSTM(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_dim,
                 hidden_dim,
                 num_layers,
                 bidir,
                 dropout,
                 cfg = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidir = bidir
        self.dropout = dropout
        # print(f"vocab_size: {vocab_size}, embed_dim: {embed_dim}, hidden_dim: {hidden_dim}, num_layers: {num_layers}, bidir: {bidir}, dropout: {dropout}")
        self.embed = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        self.lstm   = nn.LSTM(self.embed_dim, self.hidden_dim,
                             num_layers=self.num_layers,
                             bidirectional=self.bidir,
                             dropout=self.dropout,
                             batch_first=True)
        factor = 2 if self.bidir else 1
        self.fc     = nn.Linear(self.hidden_dim * factor, 1)
        self.drop   = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.drop(self.embed(x))
        _, (hidden, _) = self.lstm(x)
        # If multi-layer + bidir, take last layer’s fwd & bwd
        if hidden.shape[0] > 1:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden.squeeze(0)
        out = self.fc(self.drop(hidden))
        return out.squeeze(1)  # (B,)