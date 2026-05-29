# models/bidir_gru.py
# HW4 – Sequence Modeling | CS515 Deep Learning
#
# Part (d) – Bidirectional GRU for turning-point / buy-signal detection.
#
# This model mirrors BidirSignalLSTM but replaces LSTM cells with GRU cells.
# The bidirectional GRU processes the look-back window in both temporal
# directions and concatenates the resulting final hidden states, giving the
# classifier access to the full sequential context within the window.

import torch
import torch.nn as nn
import config


class BidirSignalGRU(nn.Module):
    """
    Bidirectional GRU for binary buy-signal classification.

    The model outputs a raw logit per sample, intended for use with
    nn.BCEWithLogitsLoss during training. Apply torch.sigmoid() at inference
    time to convert logits to buy probabilities.

    Architecture
    ------------
        Input  (batch, T, F)
            ↓
        BiGRU  [num_layers, hidden_size, bidirectional=True]
            ↓  concatenate last forward hidden h_fwd and backward hidden h_bwd
        (batch, 2 × hidden_size)
            ↓
        Dropout
            ↓
        Linear  →  (batch, 1)  →  squeeze  →  (batch,)   raw logit

    Parameters
    ----------
    input_size  : number of input features F (default: 4)
    hidden_size : hidden units per direction per GRU layer (default: 128)
    num_layers  : number of stacked BiGRU layers (default: 2)
    dropout     : dropout probability between layers and before FC
    """

    def __init__(
        self,
        input_size:  int   = config.INPUT_SIZE,         # F_hat = F + len(MA_WINDOWS)
        hidden_size: int   = config.HIDDEN_SIZE,
        num_layers:  int   = config.NUM_LAYERS,
        dropout:     float = config.DROPOUT,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(2 * hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (batch, T, F)  –  normalised OHLC look-back window

        Returns
        -------
        (batch,)  –  raw logit; apply sigmoid to obtain buy probability
        """
        _, h_n = self.gru(x)
        # h_n shape: (num_layers * 2, batch, hidden_size)
        fwd = h_n[-2]                           # (batch, hidden_size)
        bwd = h_n[-1]                           # (batch, hidden_size)
        out = torch.cat([fwd, bwd], dim=-1)     # (batch, 2 * hidden_size)
        out = self.dropout(out)
        out = self.fc(out).squeeze(-1)          # (batch,)
        return out