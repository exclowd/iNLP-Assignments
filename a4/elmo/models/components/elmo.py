import torch
from torch import nn
from torch.nn import functional as F


class CharCNN(nn.Module):
    """Character CNN"""

    def __init__(self,
                 char_embedding_dim: int = 16,
                 kernel_sizes: list = [1, 2, 3, 4, 5, 6, 7],
                 layer_sizes: list = [8, 8, 16, 32, 64, 128, 256],  # 512
                 ):
        super().__init__()
        self.output_size = sum(layer_sizes)
        self.embedding = nn.Embedding(num_embeddings=256,
                                      embedding_dim=char_embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_embedding_dim,
                      out_channels=o,
                      kernel_size=k)
            for o, k in zip(layer_sizes, kernel_sizes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, max_seq_len, max_word_len = x.size()
        x = x.view(batch_size * max_seq_len, max_word_len)

        # batch_size * max_seq_len x max_word_len ->
        # batch_size * max_seq_len x max_word_len x char_embedding_dim
        x = self.embedding(x)
        # batch_size * max_seq_len x char_embedding_dim x max_word_len
        x = x.permute(0, 2, 1)

        # batch_size * max_seq_len x char_embedding_dim x max_word_len ->
        # batch_size * max_seq_len x out_channels x max_word_len - kernel_size + 1
        # TODO - check relu
        x = [F.relu(conv(x)) for conv in self.convs]

        # batch_size * max_seq_len x out_channels x max_word_len - kernel_size + 1 ->
        # batch_size * max_seq_len x out_channels
        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in x]

        # batch_size * max_seq_len x out_channels ->
        # batch_size * max_seq_len x SUM(out_channels)
        x = torch.cat(x, dim=1)

        x = x.view(batch_size, max_seq_len, -1)
        return x


class Highway(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 num_layers: int = 2,
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_inp_size = input_size
        self.layer_out_size = input_size
        self.linear = nn.ModuleList(
            [nn.Linear(self.layer_inp_size, self.layer_out_size) for _ in range(num_layers)])
        self.gate = nn.ModuleList(
            [nn.Linear(self.layer_inp_size, self.layer_out_size) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            gate = F.sigmoid(self.gate[i](x))
            # TDOD - check relu
            linear = F.relu(self.linear[i](x))
            x = gate * linear + (1 - gate) * x
        return x


class ELMO(nn.Module):
    """
    Bidirectional Language Model to learn ELMo embeddings, as described in
    "Deep contextualized word representations"
    https://arxiv.org/abs/1802.05365

    This is a simple implementation of the ELMo model, which is a
    bidirectional LSTM with a skip connection from the input to the output.
    The output of the model is the concatenation of the forward and backward
    hidden states at each time step.
    """

    def __init__(self,
                 vocab_size: int,
                 char_embedding_dim: int = 16,
                 input_size: int = 512,
                 hidden_size: int = 256,
                 dropout: float = 0.5,
                 num_highway_layers: int = 2,
                 ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2

        self.char_cnn = CharCNN(char_embedding_dim=char_embedding_dim)
        assert(self.input_size == self.char_cnn.output_size)
        assert(self.output_size == self.char_cnn.output_size)
        self.lstm1_f = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               dropout=dropout,
                               batch_first=True)
        self.lstm1_b = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               dropout=dropout,
                               batch_first=True)
        self.lstm2_f = nn.LSTM(input_size=self.hidden_size * 2,
                               hidden_size=self.hidden_size,
                               dropout=dropout,
                               batch_first=True)
        self.lstm2_b = nn.LSTM(input_size=self.hidden_size * 2,
                               hidden_size=self.hidden_size,
                               dropout=dropout,
                               batch_first=True)
        self.highway = Highway(input_size=self.input_size,
                               num_layers=num_highway_layers)
        self.fc = nn.Linear(in_features=self.hidden_size * 2,
                            out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, max_seq_len, max_word_len = x.size()
        # batch_size x max_seq_len x max_word_len ->
        # batch_size x max_seq_len x char_cnn.output_size(512)
        x = self.char_cnn(x)
        # batch_size x max_seq_len x char_cnn.output_size(512) ->
        # batch_size x max_seq_len x input_size(512)
        x = self.highway(x)
        x_f = x
        x_b = x.flip(1)

        o1_f, (h1_f, _) = self.lstm1_f(x_f)
        o1_b, (h1_b, _) = self.lstm1_b(x_b)

        # h1 size: batch_size x max_seq_len x hidden_size(256) * 2
        h1 = torch.cat([h1_f, h1_b], dim=2).squeeze(0)

        # residual connection
        i2_f = F.relu(o1_f + x_f)
        i2_b = F.relu(o1_b + x_b)

        o2_f, (h2_f, _) = self.lstm2_f(i2_f)
        o2_b, (h2_b, _) = self.lstm2_b(i2_b)

        # h2 size: batch_size x max_seq_len x hidden_size(256) * 2
        h2 = torch.cat([h2_f, h2_b], dim=2).squeeze(0)

        out = self.fc(h2, dim=2)

        return out, x, h1, h2
