import torch
import torch.nn as nn

class SentimentSimpleRNN(nn.Module):
    """
    Vanilla (bi-)RNN for sentence-level binary sentiment.
    Embedding → (bi)RNN → Dropout → FC → logit
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int  = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 1,
                 bidir: bool     = True,
                 dropout: float  = 0.4,
                 **unused):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.rnn = nn.RNN(
            input_size   = embed_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            nonlinearity = "relu",
            bidirectional= bidir,
            dropout      = dropout if num_layers > 1 else 0.0,
            batch_first  = True,
        )

        factor = 2 if bidir else 1
        self.fc   = nn.Linear(hidden_dim * factor, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : LongTensor [batch, seq_len]
        """
        emb = self.drop(self.embed(x))          # (B, L, D)
        _, hidden = self.rnn(emb)               # hidden  [L*factor, B, D] (D: dimmension of hidden state)

        if self.rnn.bidirectional or self.rnn.num_layers > 1:
            # last layer’s fwd & bwd hidden states
            hidden = hidden[-2:] if self.rnn.bidirectional else hidden[-1:]    # [2, B, D]
            hidden = hidden.permute(1, 0, 2).reshape(x.size(0), -1)            # [B, 2*D]
        else:
            hidden = hidden.squeeze(0)          # [1, B, D] -> [B, D]
        
        # change hidden shape to [B, D]
        print(f"hidden shape: {hidden.shape}")
        logits = self.fc(self.drop(hidden))     # [B, 1]
        return logits.squeeze(1)                # [B]
