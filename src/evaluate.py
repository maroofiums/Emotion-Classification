from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BATCH_SIZE,
    MODEL_DIR,
    RANDOM_SEED,
    VOCAB_SIZE,
)
from src.dataset import load_emotion_dataset
from src.model import EmotionBiLSTM
from src.preprocessing import (
    Vocabulary,
    encode_texts,
    pad_texts_with_lengths,
)
from src.visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
)


MAX_LENGTH = 50
REPORT_DIR = Path("reports")


def load_trained_model(
    checkpoint_path,
    device,
):
    """
    Load the trained model and vocabulary.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    vocabulary = Vocabulary(
        max_size=VOCAB_SIZE
    )

    vocabulary.token_to_id = checkpoint[
        "vocabulary"
    ]

    vocabulary.id_to_token = {
        idx: token
        for token, idx
        in vocabulary.token_to_id.items()
    }

    model = EmotionBiLSTM(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, vocabulary


def prepare_test_data(
    dataset,
    vocabulary,
):
    """
    Prepare the test split for inference.
    """

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    encoded = encode_texts(
        texts=test_texts,
        vocabulary=vocabulary,
        max_length=MAX_LENGTH,
    )

    inputs, lengths = pad_texts_with_lengths(
        encoded
    )

    labels = torch.tensor(
        test_labels,
        dtype=torch.long,
    )

    return (
        inputs,
        lengths,
        labels,
        test_texts,
    )


def predict(
    model,
    inputs,
    lengths,
    labels,
    device,
):
    """
    Generate predictions for the test set.
    """

    test_dataset = TensorDataset(
        inputs,
        lengths,
        labels,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for (
            batch_inputs,
            batch_lengths,
            batch_labels,
        ) in test_loader:

            batch_inputs = batch_inputs.to(
                device
            )

            batch_lengths = batch_lengths.to(
                device
            )

            batch_labels = batch_labels.to(
                device
            )

            outputs = model(
                batch_inputs,
                batch_lengths,
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                batch_labels.cpu().tolist()
            )

    return all_labels, all_predictions


def evaluate_model(
    y_true,
    y_pred,
):
    """
    Calculate and print overall evaluation metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\nTest Metrics")
    print("=" * 50)

    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"Weighted F1  : {f1:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "weighted_f1": f1,
        "macro_f1": macro_f1,
    }


def print_classification_report(
    y_true,
    y_pred,
    label_names,
):
    """
    Print per-class evaluation metrics.
    """

    print("\nClassification Report")
    print("=" * 70)

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )

    print(report)


def print_confusion_matrix(
    y_true,
    y_pred,
    label_names,
):
    """
    Print the numerical confusion matrix.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    print("\nConfusion Matrix")
    print("=" * 50)

    print(matrix)

    print("\nLabels:")
    print(label_names)

    return matrix


def show_errors(
    texts,
    y_true,
    y_pred,
    label_names,
    num_examples=20,
):
    """
    Display misclassified examples.
    """

    print("\nMisclassified Examples")
    print("=" * 70)

    count = 0

    for (
        text,
        true_label,
        predicted_label,
    ) in zip(
        texts,
        y_true,
        y_pred,
    ):

        if true_label == predicted_label:
            continue

        print(f"\nText: {text}")

        print(
            f"Actual: "
            f"{label_names[true_label]}"
        )

        print(
            f"Predicted: "
            f"{label_names[predicted_label]}"
        )

        count += 1

        if count >= num_examples:
            break


def main():

    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # Load dataset
    dataset = load_emotion_dataset()

    # Load model
    model_path = (
        MODEL_DIR
        / "emotion_bilstm.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run training first:\n"
            "python -m src.train"
        )

    model, vocabulary = (
        load_trained_model(
            model_path,
            device,
        )
    )

    print(
        f"Vocabulary size: "
        f"{len(vocabulary)}"
    )

    # Prepare test data
    (
        test_inputs,
        test_lengths,
        test_labels,
        test_texts,
    ) = prepare_test_data(
        dataset,
        vocabulary,
    )

    # Predict
    y_true, y_pred = predict(
        model=model,
        inputs=test_inputs,
        lengths=test_lengths,
        labels=test_labels,
        device=device,
    )

    # Labels
    label_names = dataset[
        "train"
    ].features["label"].names

    # Metrics
    evaluate_model(
        y_true,
        y_pred,
    )

    # Classification report
    print_classification_report(
        y_true,
        y_pred,
        label_names,
    )

    # Confusion matrix
    print_confusion_matrix(
        y_true,
        y_pred,
        label_names,
    )

    # Error analysis
    show_errors(
        texts=test_texts,
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    # Visualizations
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
        save_path=(
            REPORT_DIR
            / "confusion_matrix.png"
        ),
    )

    plot_class_distribution(
        y_true=y_true,
        label_names=label_names,
        save_path=(
            REPORT_DIR
            / "class_distribution.png"
        ),
    )


if __name__ == "__main__":
    main()