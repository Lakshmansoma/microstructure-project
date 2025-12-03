import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GrainLSTM(nn.Module):
  

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # (B, T, F)
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # scalar stress
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
       
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        packed_out, (h_n, c_n) = self.lstm(packed)
      
        num_directions = 2 if self.bidirectional else 1
        h_n = h_n.view(self.num_layers, num_directions, x.size(0), self.hidden_dim)
        last_layer_h = h_n[-1]  # (num_directions, B, hidden_dim)

        if self.bidirectional:
            h_cat = torch.cat([last_layer_h[0], last_layer_h[1]], dim=1)  # (B, 2H)
        else:
            h_cat = last_layer_h[0]  # (B, H)

        y_hat = self.fc(h_cat)  # (B, 1)
        return y_hat
