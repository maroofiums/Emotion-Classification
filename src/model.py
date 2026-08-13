import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class EmotionBiLSTM(nn.Module):
    """
    Bidirectional LSTM for emotion classification.
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

        # -------------------------
        # Embedding
        # -------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # -------------------------
        # BiLSTM
        # -------------------------

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        # -------------------------
        # Dropout
        # -------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        # -------------------------
        # Classifier
        # -------------------------

        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                [batch_size, sequence_length]

            lengths:
                [batch_size]

        Returns:
            logits:
                [batch_size, num_classes]
        """

        # -------------------------
        # Embedding
        # -------------------------

        embedded = self.embedding(x)

        # Shape:
        # [batch, seq_len, embedding_dim]

        # -------------------------
        # Pack padded sequences
        # -------------------------

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # -------------------------
        # BiLSTM
        # -------------------------

        _, (hidden, _) = self.lstm(
            packed
        )

        # hidden shape:
        #
        # [num_layers * 2,
        #  batch_size,
        #  hidden_dim]

        # -------------------------
        # Final forward hidden state
        # -------------------------

        forward_hidden = hidden[-2]

        # Shape:
        # [batch_size, hidden_dim]

        # -------------------------
        # Final backward hidden state
        # -------------------------

        backward_hidden = hidden[-1]

        # Shape:
        # [batch_size, hidden_dim]

        # -------------------------
        # Combine directions
        # -------------------------

        hidden_state = torch.cat(
            (
                forward_hidden,
                backward_hidden,
            ),
            dim=1,
        )

        # Shape:
        # [batch_size, hidden_dim * 2]

        # -------------------------
        # Dropout
        # -------------------------

        hidden_state = self.dropout(
            hidden_state
        )

        # -------------------------
        # Classification
        # -------------------------

        logits = self.classifier(
            hidden_state
        )

        # Shape:
        # [batch_size, num_classes]

        return logits


if __name__ == "__main__":

    model = EmotionBiLSTM(
        vocab_size=20_000,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=6,
        dropout=0.3,
    )

    # Fake batch
    x = torch.randint(
        low=0,
        high=20_000,
        size=(4, 20),
    )

    lengths = torch.tensor(
        [20, 17, 12, 8]
    )

    output = model(
        x,
        lengths,
    )

    print("Input shape:")
    print(x.shape)

    print("\nLengths:")
    print(lengths)

    print("\nOutput shape:")
    print(output.shape)

    print("\nModel:")
    print(model)