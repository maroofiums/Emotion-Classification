from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from src.config import (
    DROPOUT,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    MODEL_DIR,
    NUM_CLASSES,
    NUM_LAYERS,
    VOCAB_SIZE,
)
from src.model import EmotionBiLSTM
from src.preprocessing import (
    Vocabulary,
    encode_texts,
    pad_texts_with_lengths,
)


MAX_LENGTH = 50

LABEL_NAMES = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]


class EmotionPredictor:
    """
    Handles loading the trained model and
    making predictions on new text.
    """

    def __init__(
        self,
        model_path: str | Path = (
            MODEL_DIR / "emotion_bilstm.pt"
        ),
        device: str | None = None,
    ):
        self.model_path = Path(model_path)

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(
                device
            )

        self.model = None
        self.vocabulary = None

        self._load_model()

    def _load_model(self):
        """
        Load model checkpoint and vocabulary.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}\n"
                "Run training first:\n"
                "python -m src.train"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        # Rebuild vocabulary
        self.vocabulary = Vocabulary(
            max_size=VOCAB_SIZE
        )

        self.vocabulary.token_to_id = (
            checkpoint["vocabulary"]
        )

        self.vocabulary.id_to_token = {
            idx: token
            for token, idx
            in self.vocabulary.token_to_id.items()
        }

        # Rebuild model
        self.model = EmotionBiLSTM(
            vocab_size=checkpoint["vocab_size"],
            embedding_dim=checkpoint[
                "embedding_dim"
            ],
            hidden_dim=checkpoint[
                "hidden_dim"
            ],
            num_layers=checkpoint[
                "num_layers"
            ],
            num_classes=checkpoint[
                "num_classes"
            ],
            dropout=checkpoint["dropout"],
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    def predict(
        self,
        text: str,
    ) -> Dict:
        """
        Predict emotion for a single text.
        """

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        # Tokenize + encode
        encoded = encode_texts(
            texts=[text],
            vocabulary=self.vocabulary,
            max_length=MAX_LENGTH,
        )

        # Padding
        inputs, lengths = (
            pad_texts_with_lengths(
                encoded
            )
        )

        inputs = inputs.to(self.device)
        lengths = lengths.to(self.device)

        # Model inference
        with torch.no_grad():

            logits = self.model(
                inputs,
                lengths,
            )

            probabilities = F.softmax(
                logits,
                dim=1,
            )

        # Prediction
        predicted_id = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_id,
        ].item()

        emotion = LABEL_NAMES[
            predicted_id
        ]

        # All probabilities
        probability_dict = {
            LABEL_NAMES[index]: float(
                probabilities[0, index]
            )
            for index in range(
                len(LABEL_NAMES)
            )
        }

        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probability_dict,
        }


def predict_emotion(
    text: str,
) -> Dict:
    """
    Convenience function for making
    a single prediction.
    """

    predictor = EmotionPredictor()

    return predictor.predict(text)


if __name__ == "__main__":

    predictor = EmotionPredictor()

    examples = [
        "I finally got the job!",
        "I am terrified about what happens next.",
        "I miss my family so much.",
        "This makes me incredibly angry.",
        "I love spending time with you.",
        "Wow, I never expected that!",
    ]

    for text in examples:

        result = predictor.predict(
            text
        )

        print("\nText:")
        print(text)

        print(
            f"Emotion: "
            f"{result['emotion']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.2%}"
        )

        print(
            "\nProbabilities:"
        )

        for (
            emotion,
            probability,
        ) in result[
            "probabilities"
        ].items():

            print(
                f"  {emotion:<10}"
                f"{probability:.2%}"
            )