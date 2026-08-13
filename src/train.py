import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from src.config import (
    BATCH_SIZE,
    DROPOUT,
    EMBEDDING_DIM,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_CLASSES,
    NUM_LAYERS,
    RANDOM_SEED,
    VOCAB_SIZE,
)

from src.dataset import (
    load_emotion_dataset,
)

from src.model import (
    EmotionBiLSTM,
)

from src.preprocessing import (
    Vocabulary,
    encode_texts,
    pad_texts_with_lengths,
)


MAX_LENGTH = 50


def set_seed(
    seed: int = RANDOM_SEED,
) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(dataset):
    """
    Prepare train and validation data.

    Vocabulary is built ONLY from
    the training set.
    """

    # -------------------------
    # Get raw data
    # -------------------------

    train_texts = dataset["train"]["text"]

    val_texts = dataset["validation"]["text"]

    train_labels = dataset["train"]["label"]

    val_labels = dataset["validation"]["label"]

    # -------------------------
    # Build vocabulary
    # -------------------------

    vocabulary = Vocabulary(
        max_size=VOCAB_SIZE
    )

    vocabulary.build(
        train_texts
    )

    # -------------------------
    # Encode
    # -------------------------

    train_encoded = encode_texts(
        texts=train_texts,
        vocabulary=vocabulary,
        max_length=MAX_LENGTH,
    )

    val_encoded = encode_texts(
        texts=val_texts,
        vocabulary=vocabulary,
        max_length=MAX_LENGTH,
    )

    # -------------------------
    # Padding + lengths
    # -------------------------

    (
        train_inputs,
        train_lengths,
    ) = pad_texts_with_lengths(
        train_encoded
    )

    (
        val_inputs,
        val_lengths,
    ) = pad_texts_with_lengths(
        val_encoded
    )

    # -------------------------
    # Labels
    # -------------------------

    train_labels = torch.tensor(
        train_labels,
        dtype=torch.long,
    )

    val_labels = torch.tensor(
        val_labels,
        dtype=torch.long,
    )

    return (
        vocabulary,
        train_inputs,
        train_lengths,
        train_labels,
        val_inputs,
        val_lengths,
        val_labels,
    )


def create_dataloaders(
    train_inputs,
    train_lengths,
    train_labels,
    val_inputs,
    val_lengths,
    val_labels,
):
    """
    Create PyTorch DataLoaders.
    """

    train_dataset = TensorDataset(
        train_inputs,
        train_lengths,
        train_labels,
    )

    val_dataset = TensorDataset(
        val_inputs,
        val_lengths,
        val_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Train model for one epoch.
    """

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    for (
        inputs,
        lengths,
        labels,
    ) in loader:

        # -------------------------
        # Move data to device
        # -------------------------

        inputs = inputs.to(device)

        lengths = lengths.to(device)

        labels = labels.to(device)

        # -------------------------
        # Clear gradients
        # -------------------------

        optimizer.zero_grad()

        # -------------------------
        # Forward pass
        # -------------------------

        outputs = model(
            inputs,
            lengths,
        )

        # -------------------------
        # Calculate loss
        # -------------------------

        loss = criterion(
            outputs,
            labels,
        )

        # -------------------------
        # Backpropagation
        # -------------------------

        loss.backward()

        # -------------------------
        # Update parameters
        # -------------------------

        optimizer.step()

        # -------------------------
        # Statistics
        # -------------------------

        total_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        total_loss / len(loader)
    )

    accuracy = correct / total

    return (
        average_loss,
        accuracy,
    )


def validate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate model on validation data.
    """

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for (
            inputs,
            lengths,
            labels,
        ) in loader:

            inputs = inputs.to(device)

            lengths = lengths.to(device)

            labels = labels.to(device)

            # Forward pass
            outputs = model(
                inputs,
                lengths,
            )

            # Loss
            loss = criterion(
                outputs,
                labels,
            )

            total_loss += loss.item()

            # Predictions
            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / len(loader)
    )

    accuracy = correct / total

    return (
        average_loss,
        accuracy,
    )


def main():

    # =========================
    # Setup
    # =========================

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # =========================
    # Load Dataset
    # =========================

    print(
        "\nLoading dataset..."
    )

    dataset = load_emotion_dataset()

    # =========================
    # Prepare Data
    # =========================

    print(
        "Preparing data..."
    )

    (
        vocabulary,
        train_inputs,
        train_lengths,
        train_labels,
        val_inputs,
        val_lengths,
        val_labels,
    ) = prepare_data(
        dataset
    )

    print(
        f"Vocabulary size: "
        f"{len(vocabulary)}"
    )

    print(
        f"Train samples: "
        f"{len(train_labels)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_labels)}"
    )

    # =========================
    # DataLoaders
    # =========================

    train_loader, val_loader = (
        create_dataloaders(
            train_inputs,
            train_lengths,
            train_labels,
            val_inputs,
            val_lengths,
            val_labels,
        )
    )

    print(
        f"Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    # =========================
    # Model
    # =========================

    model = EmotionBiLSTM(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    print(
        "\nModel created."
    )

    # =========================
    # Loss Function
    # =========================

    criterion = nn.CrossEntropyLoss()

    # =========================
    # Optimizer
    # =========================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # =========================
    # Model Directory
    # =========================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================
    # Training
    # =========================

    best_val_accuracy = 0.0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        val_loss, val_accuracy = (
            validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
        )

        print(
            f"\nEpoch "
            f"{epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss:   "
            f"{val_loss:.4f} | "
            f"Val Acc:    "
            f"{val_accuracy:.4f}"
        )

        # =====================
        # Save Best Model
        # =====================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            model_path = (
                MODEL_DIR
                / "emotion_bilstm.pt"
            )

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "vocab_size":
                    len(vocabulary),

                "embedding_dim":
                    EMBEDDING_DIM,

                "hidden_dim":
                    HIDDEN_DIM,

                "num_layers":
                    NUM_LAYERS,

                "num_classes":
                    NUM_CLASSES,

                "dropout":
                    DROPOUT,

                "max_length":
                    MAX_LENGTH,

                "vocabulary":
                    vocabulary.token_to_id,
            }

            torch.save(
                checkpoint,
                model_path,
            )

            print(
                f"Saved best model → "
                f"{model_path}"
            )

    print(
        "\nTraining complete."
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()