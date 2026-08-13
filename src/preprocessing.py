import re
from collections import Counter
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence



def tokenize(text: str) -> List[str]:
    """
    Convert text into lowercase word tokens.
    """

    text = text.lower()

    tokens = re.findall(
        r"\b\w+\b",
        text,
    )

    return tokens


class Vocabulary:
    """
    Maps tokens <-> integer IDs.

    0 -> <PAD>
    1 -> <UNK>
    2+ -> actual vocabulary tokens
    """

    def __init__(self, max_size: int = 20_000):
        self.max_size = max_size

        self.token_to_id = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
        }

        self.id_to_token = {
            PAD_IDX: PAD_TOKEN,
            UNK_IDX: UNK_TOKEN,
        }

    def build(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.
        """

        counter = Counter()

        for text in texts:
            tokens = tokenize(text)
            counter.update(tokens)

        # Reserve two IDs for PAD and UNK
        most_common = counter.most_common(
            self.max_size - 2
        )

        for idx, (token, _) in enumerate(
            most_common,
            start=2,
        ):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def encode(self, text: str) -> List[int]:
        """
        Convert text into token IDs.
        """

        tokens = tokenize(text)

        return [
            self.token_to_id.get(
                token,
                UNK_IDX,
            )
            for token in tokens
        ]

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back into text.
        """

        tokens = [
            self.id_to_token.get(
                idx,
                UNK_TOKEN,
            )
            for idx in ids
        ]

        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.token_to_id)


def encode_texts(
    texts: List[str],
    vocabulary: Vocabulary,
    max_length: int,
) -> List[torch.Tensor]:
    """
    Encode multiple texts.

    Sequences longer than max_length are truncated.
    """

    encoded_texts = []

    for text in texts:

        ids = vocabulary.encode(text)

        # Prevent empty sequences
        if len(ids) == 0:
            ids = [UNK_IDX]

        ids = ids[:max_length]

        encoded_texts.append(
            torch.tensor(
                ids,
                dtype=torch.long,
            )
        )

    return encoded_texts


def pad_texts(
    encoded_texts: List[torch.Tensor],
) -> torch.Tensor:
    """
    Pad sequences to the same length.

    Returns:
        Tensor of shape:
        [batch_size, sequence_length]
    """

    return pad_sequence(
        encoded_texts,
        batch_first=True,
        padding_value=PAD_IDX,
    )


def pad_texts_with_lengths(
    encoded_texts: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences and preserve their original lengths.

    Returns:
        padded_texts:
            [batch_size, sequence_length]

        lengths:
            [batch_size]
    """

    lengths = torch.tensor(
        [
            len(sequence)
            for sequence in encoded_texts
        ],
        dtype=torch.long,
    )

    padded_texts = pad_sequence(
        encoded_texts,
        batch_first=True,
        padding_value=PAD_IDX,
    )

    return padded_texts, lengths


def preprocess_batch(
    texts: List[str],
    vocabulary: Vocabulary,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complete preprocessing pipeline for a batch.
    """

    encoded_texts = encode_texts(
        texts=texts,
        vocabulary=vocabulary,
        max_length=max_length,
    )

    padded_texts, lengths = (
        pad_texts_with_lengths(
            encoded_texts
        )
    )

    return padded_texts, lengths


if __name__ == "__main__":

    texts = [
        "I feel very happy today!",
        "I am extremely sad.",
        "I am angry about this.",
    ]

    vocabulary = Vocabulary(
        max_size=100
    )

    vocabulary.build(texts)

    print("Vocabulary size:")
    print(len(vocabulary))

    print("\nVocabulary:")
    print(vocabulary.token_to_id)

    encoded = encode_texts(
        texts,
        vocabulary,
        max_length=10,
    )

    print("\nEncoded:")
    print(encoded)

    padded, lengths = (
        pad_texts_with_lengths(
            encoded
        )
    )

    print("\nPadded:")
    print(padded)

    print("\nLengths:")
    print(lengths)

    print("\nShape:")
    print(padded.shape)