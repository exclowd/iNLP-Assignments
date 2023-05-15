import torch
import torch.nn as nn
from torch.nn import functional as F

from elmo.models.components.elmo import ELMO


class ELMOEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 glove_embedding_dim: int = 300,
                 char_embedding_dim: int = 16,
                 elmo_input_size: int = 512,
                 elmo_hidden_size: int = 256,
                 elmo_dropout: float = 0.5,
                 elmo_num_highway_layers: int = 2,
                 ):
        super().__init__()

        self.elmo = ELMO(vocab_size=vocab_size,
                         char_embedding_dim=char_embedding_dim,
                         input_size=elmo_input_size,
                         hidden_size=elmo_hidden_size,
                         dropout=elmo_dropout,
                         num_highway_layers=elmo_num_highway_layers)

        self.elmo_size = self.elmo.output_size

        self.elmo_fc = nn.Linear(in_features=3, out_features=1, bias=False)

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=glove_embedding_dim)

        self.embedding_size = self.elmo.output_size + glove_embedding_dim


    def forward(self, char: torch.Tensor, word: torch.Tensor) -> torch.Tensor:
        batch_size, max_seq_len, max_word_len = char.size()

        _, x, h1, h2 = self.elmo(char)
        x = torch.cat([x, h1, h2], dim=1)
        x = self.elmo_fc(x).squeeze(-1)

        word = self.embedding(word)



class Classifier(nn.Module):
    def __init__(self,
                 elmo: nn.Module,
                 vocab_size: int,
                 out_features: int = 3,
                 hidden_size: int = 512,
                 char_embedding_dim: int = 16,
                 glove_embedding_dim: int = 300,
                 input_size: int = 256,
                 elmo_hidden_size: int = 256,
                 dropout: float = 0.5,
                 highway_layers: int = 2,
                 ):
        super().__init__()

        self.elmo_linear = nn.Linear(in_features=3, out_features=1, bias=False)

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            batch_first=True)

        self.classifier = nn.Linear(in_features=hidden_size,
                                    out_features=out_features)
        # freeze elmo
        for param in self.elmo.parameters():
            param.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        # batch_size x max_seq_len x 3 x char_embedding_dim(512) ->
        # batch_size x max_seq_len x 1 x input_size(512)
        x = self.elmo_linear(x).squeeze(-1)

        # batch_size x max_seq_len x hidden_size(512)
        x, _ = self.lstm(x)

        # batch_size x hidden_size(512)
        x = x[:, -1, :]

        # batch_size x out_features(3)
        x = F.softmax(self.classifier(x) , dim=1)

        return x
