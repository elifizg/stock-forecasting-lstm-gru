# models/gru.py
# HW4 – Sequence Modeling | CS515 Deep Learning
#
# Part (b) & (c) – Stacked GRU for stock return forecasting.
#
# The GRU (Gated Recurrent Unit) is a simplified recurrent architecture that
# merges the cell and hidden states into a single hidden state and uses only
# two gates (reset and update) instead of three. GRUs typically have fewer
# parameters than LSTMs, train faster, and perform comparably on many
# sequence modelling tasks.

import torch
import torch.nn as nn
import config


class StockGRU(nn.Module):
    """
    Stacked GRU network for multi-horizon stock return forecasting.

    Architecture
    ------------
        Input  (batch, T, F)
            ↓
        GRU  [num_layers stacked, hidden_size, dropout between layers]
            ↓  last hidden state h_T of the final layer
        Dropout
            ↓
        Linear  →  (batch, D)   D = 5 return predictions

    The GRU update equations at each time step t are:

        z_t = σ(W_z [h_{t-1}, x_t] + b_z)              # update gate
        r_t = σ(W_r [h_{t-1}, x_t] + b_r)              # reset gate
        h̃_t = tanh(W_h [r_t ⊙ h_{t-1}, x_t] + b_h)   # candidate hidden state
        h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t        # hidden state update

    The update gate z_t controls how much of the previous hidden state is
    retained, while the reset gate r_t controls how much past information is
    forgotten when computing the candidate state.

    Parameters
    ----------
    input_size  : number of input features F (default: 4)
    hidden_size : number of hidden units per GRU layer (default: 128)
    num_layers  : number of stacked GRU layers (default: 2)
    dropout     : dropout probability applied between layers and before FC
    output_size : number of forecast horizons D (default: 5)
    """

    def __init__(
        self,
        input_size:  int   = config.INPUT_SIZE,         # F_hat = F + len(MA_WINDOWS)
        hidden_size: int   = config.HIDDEN_SIZE,
        num_layers:  int   = config.NUM_LAYERS,
        dropout:     float = config.DROPOUT,
        output_size: int   = config.HORIZON,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (batch, T, F)  –  normalised OHLC look-back window

        Returns
        -------
        (batch, D)  –  predicted return ratios for d = 1, …, D
        """
        # gru returns (output, h_n):
        #   h_n : (num_layers, batch, hidden)  – final hidden states
        _, h_n = self.gru(x)

        out = h_n[-1]          # last layer's hidden state  (batch, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)     # (batch, D)
        return out