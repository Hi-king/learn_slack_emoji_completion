import math
from unicodedata import bidirectional
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    r"""https://github.com/pytorch/examples/blob/2bf23f105237e03ee2501f29670fb6a9ca915096/word_language_model/model.py#L65
    
    Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SimpleLSTM(nn.Module):
    def __init__(
        self,
        n_input=100,
        n_token=26,
        hidden_dim=100,
        num_layers=3,
        num_head=20,
        dropout=0.5,
        positional_encoding=True,
        max_len=100,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.hidden_dim = hidden_dim
        self.input_encoder = nn.Embedding(
            num_embeddings=n_token,
            embedding_dim=self.n_input)
        self.lstm = nn.LSTM(
            self.n_input, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout,
            bidirectional=True,
        )
        self.decoder = nn.Linear(hidden_dim*2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_encoder(x)
        x, _ = self.lstm(x)
        x = self.decoder(torch.cat((x[0, :, :self.hidden_dim], x[-1, :, :self.hidden_dim]), 1))
        return x



class Transformer(nn.Module):

    def __init__(
        self,
        n_input=100,
        n_token=26,
        num_layers=3,
        num_head=20,
        dropout=0.5,
        positional_encoding=True,
        max_len=100,
    ) -> None:
        super().__init__()

        self.positional_encoding = positional_encoding
        self.n_input = n_input
        if self.positional_encoding:
            self.pos_encoder = PositionalEncoding(self.n_input, dropout, max_len=max_len)
        self.input_encoder = nn.Embedding(num_embeddings=n_token,
                                          embedding_dim=self.n_input)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.n_input, nhead=num_head),
            num_layers=num_layers)
        self.decoder = nn.Linear(self.n_input*2, 1)

    def forward(self, x):
        x = self.input_encoder(x) * math.sqrt(self.n_input)
        if self.positional_encoding:
            x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(torch.cat((x[0], x[-1]), 1))
        # x = self.decoder(x[0])
        return x
