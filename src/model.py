import torch
import torch.nn as nn


class EmotionBiLSTM(nn.Module):
    """
    Bidirectional LSTM model for emotion classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # x:
        # [batch_size, sequence_length]

        embedded = self.embedding(x)

        # embedded:
        # [batch_size, sequence_length, embedding_dim]

        output, (hidden, cell) = self.lstm(embedded)

        # Because the LSTM is bidirectional:
        #
        # hidden shape:
        # [num_layers * 2, batch_size, hidden_dim]

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        # Concatenate forward and backward representations
        #
        # [batch_size, hidden_dim * 2]

        hidden_state = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        hidden_state = self.dropout(
            hidden_state
        )

        logits = self.classifier(
            hidden_state
        )

        return logits