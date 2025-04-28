import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,
                 max_len: int,
                 d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # register as buffer so it follows the model’s device
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1)] # x: (B, T, D), pe: (T, D)
        return x

class SentimentTransformer(nn.Module):
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 d_model: int,
                 ff_hidden: int,
                 vocab_size: int,   # for embedding
                 max_len: int,      # max toks in sentence, for positional encoding
                 activation: str = "relu",
                 pool: str = "mean",      # "cls", "mean", "max"
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.possitional_encoding = PositionalEncoding(max_len, d_model)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_hidden,
                activation=activation,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.pool = pool
        if self.pool == "cls":                # need a learned [CLS] token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x : (B, T, D)
        Output: logits [B]
        """
        emb = self.embedding(x)                          # (B, T, D)
        if self.pool == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1)   # (B, 1, D))
            
        emb = self.possitional_encoding(emb)
        emb = self.dropout(emb)
        
        if self.pool == "cls":
            emb = torch.cat([cls, emb], dim=1)       # prepend

        # key_padding_mask: true at <pad> tokens
        pad_mask = (x == 0)
        if self.pool == "cls":
            pad_mask = torch.cat([torch.zeros_like(pad_mask[:, :1]), pad_mask], dim=1)

        enc_out = self.encoder(emb, src_key_padding_mask=pad_mask)

        if self.pool == "cls":
            sent_vec = enc_out[:, 0]                 # first token
        elif self.pool == "max":
            sent_vec, _ = enc_out.max(dim=1)
        else:                                        # default "mean"
            lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1)
            sent_vec = enc_out.sum(dim=1) / lengths  # masked mean

        logits = self.fc(self.dropout(sent_vec))
        return logits.squeeze(1)
        