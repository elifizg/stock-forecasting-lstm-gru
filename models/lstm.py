# models/lstm.py
# HW4 – Sequence Modeling | CS515 Deep Learning
#
# Part (b) & (c) – Stacked LSTM for stock return forecasting.
#
# The LSTM (Long Short-Term Memory) is a recurrent neural network architecture
# designed to capture long-range temporal dependencies. Unlike vanilla RNNs,
# LSTMs use a gating mechanism (forget, input, and output gates) and a separate
# cell state to selectively retain or discard information over time. This makes
# them well-suited for financial time series, where long-range patterns (e.g.,
# quarterly trends) coexist with short-range noise.

import torch
import torch.nn as nn
import config


class StockLSTM(nn.Module):
    """
    Stacked LSTM network for multi-horizon stock return forecasting.

    Architecture
    ------------
        Input  (batch, T, F)
            ↓
        LSTM  [num_layers stacked, hidden_size, dropout between layers]
            ↓  last hidden state h_T of the final layer
        Dropout
            ↓
        Linear  →  (batch, D)   D = 5 return predictions

    The LSTM update equations at each time step t are:

        f_t = σ(W_f [h_{t-1}, x_t] + b_f)      # forget gate
        i_t = σ(W_i [h_{t-1}, x_t] + b_i)      # input gate
        c̃_t = tanh(W_c [h_{t-1}, x_t] + b_c)  # candidate cell state
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t      # cell state update
        o_t = σ(W_o [h_{t-1}, x_t] + b_o)      # output gate
        h_t = o_t ⊙ tanh(c_t)                  # hidden state

    Only the final hidden state h_T (after processing all T time steps) is
    passed to the output layer, as it summarises the entire look-back window.

    Parameters
    ----------
    input_size  : number of input features F (default: 4 – Open, High, Low, Close)
    hidden_size : number of hidden units per LSTM layer (default: 128)
    num_layers  : number of stacked LSTM layers (default: 2)
    dropout     : dropout probability applied between LSTM layers and before FC
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

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            # Dropout is applied between LSTM layers (not after the last layer).
            # Disabled for single-layer models to avoid a PyTorch warning.
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
        # lstm returns (output, (h_n, c_n)):
        #   output : (batch, T, hidden)  – hidden state at every time step
        #   h_n    : (num_layers, batch, hidden)  – final hidden states
        _, (h_n, _) = self.lstm(x)

        # Take the hidden state of the last LSTM layer as the sequence summary.
        out = h_n[-1]          # (batch, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)     # (batch, D)
        return out