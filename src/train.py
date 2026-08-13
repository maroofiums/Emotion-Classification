import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

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
from src.dataset import load_emotion_dataset
from src.model import EmotionBiLSTM
from src.preprocessing import Vocabulary, encode_texts, pad_texts


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(
    dataset: Any,
) -> tuple[
    Vocabulary,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    """
    Build vocabulary using only the training set
    and convert all splits into tensors.
    """

    train_texts = dataset["train"]["text"]
    val_texts = dataset["validation"]["text"]

    train_labels = dataset["train"]["label"]
    val_labels = dataset["validation"]["label"]

    # Build vocabulary ONLY from training data
    vocabulary = Vocabulary(
        max_size=VOCAB_SIZE,
    )

    vocabulary.build(train_texts)

    # Encode
    train_encoded = encode_texts(
        train_texts,
        vocabulary,
        max_length=50,
    )

    val_encoded = encode_texts(
        val_texts,
        vocabulary,
        max_length=50,
    )

    # Pad
    train_inputs = pad_texts(train_encoded)
    val_inputs = pad_texts(val_encoded)

    # Labels
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
        train_labels,
        val_inputs,
        val_labels,
    )


def create_dataloaders(
    train_inputs: Tensor,
    train_labels: Tensor,
    val_inputs: Tensor,
    val_labels: Tensor,
) -> tuple[
    DataLoader[TensorDataset],
    DataLoader[TensorDataset],
]:
    """Create PyTorch DataLoaders."""

    train_dataset = TensorDataset(
        train_inputs,
        train_labels,
    )

    val_dataset = TensorDataset(
        val_inputs,
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

    return train_loader, val_loader


def train_one_epoch(
    model: EmotionBiLSTM,
    loader: DataLoader[TensorDataset],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train model for one epoch."""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        total_loss / len(loader)
    )

    accuracy = correct / total

    return average_loss, accuracy


def validate(
    model: EmotionBiLSTM,
    loader: DataLoader[TensorDataset],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on validation data."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / len(loader)
    )

    accuracy = correct / total

    return average_loss, accuracy


def main() -> None:
    """Load data, train the model, and save the best checkpoint."""

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # Load dataset
    dataset = load_emotion_dataset()

    # Prepare data
    (
        vocabulary,
        train_inputs,
        train_labels,
        val_inputs,
        val_labels,
    ) = prepare_data(dataset)

    print(
        f"Vocabulary size: {len(vocabulary)}"
    )

    print(
        f"Train samples: {len(train_labels)}"
    )

    print(
        f"Validation samples: {len(val_labels)}"
    )

    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_inputs,
        train_labels,
        val_inputs,
        val_labels,
    )

    # Model
    model = EmotionBiLSTM(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    print("\nModel:")
    print(model)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Training
    best_val_accuracy = 0.0

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:   {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            model_path = (
                MODEL_DIR
                / "emotion_bilstm.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": len(vocabulary),
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "num_classes": NUM_CLASSES,
                    "dropout": DROPOUT,
                    "vocabulary": vocabulary.token_to_id,
                },
                model_path,
            )

            print(
                f"Saved best model → {model_path}"
            )


if __name__ == "__main__":
    main()