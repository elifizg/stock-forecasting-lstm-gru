# models/bidir_lstm.py
# HW4 – Sequence Modeling | CS515 Deep Learning
#
# Part (d) – Bidirectional LSTM for turning-point / buy-signal detection.
#
# A bidirectional LSTM processes the input sequence in both forward and
# backward directions and concatenates the resulting hidden states. This
# allows the model to capture context from both past and future time steps
# within the look-back window, which can be beneficial for detecting
# structural turning points in a price series.
#
# Note: bidirectionality is applied within the T-step look-back window only;
# future prices beyond the window are never observed during inference.

import torch
import torch.nn as nn
import config


class BidirSignalLSTM(nn.Module):
    """
    Bidirectional LSTM for binary buy-signal classification.

    The model outputs a raw logit (un-sigmoided scalar) per sample. During
    training this is paired with nn.BCEWithLogitsLoss, which applies the
    sigmoid internally for numerical stability. At inference time, applying
    torch.sigmoid() to the output gives the buy probability.

    Architecture
    ------------
        Input  (batch, T, F)
            ↓
        BiLSTM  [num_layers, hidden_size, bidirectional=True]
            ↓  concatenate last forward hidden h_fwd and backward hidden h_bwd
        (batch, 2 × hidden_size)
            ↓
        Dropout
            ↓
        Linear  →  (batch, 1)  →  squeeze  →  (batch,)   raw logit

    Because the LSTM is bidirectional, the effective hidden dimension doubles.
    The final hidden states of the topmost forward and backward layers are
    concatenated to form the sequence representation passed to the classifier.

    Parameters
    ----------
    input_size  : number of input features F (default: 4)
    hidden_size : hidden units per direction per LSTM layer (default: 128)
    num_layers  : number of stacked BiLSTM layers (default: 2)
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

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # The FC input is 2 × hidden_size because forward and backward
        # hidden states are concatenated along the feature dimension.
        self.fc = nn.Linear(2 * hidden_size, 1)

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
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * 2, batch, hidden_size)
        # For the topmost layer:
        #   h_n[-2] = final forward  hidden state
        #   h_n[-1] = final backward hidden state
        fwd = h_n[-2]                           # (batch, hidden_size)
        bwd = h_n[-1]                           # (batch, hidden_size)
        out = torch.cat([fwd, bwd], dim=-1)     # (batch, 2 * hidden_size)
        out = self.dropout(out)
        out = self.fc(out).squeeze(-1)          # (batch,)
        return out