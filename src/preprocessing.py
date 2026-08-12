import re
from collections import Counter
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from src.config import PAD_TOKEN, UNK_TOKEN 

class Vocabulary:
    def __init__(self, max_size: int = 20_000):
        self.max_size = max_size
        self.token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1
        }
        self.id_to_token = {
            0: PAD_TOKEN, 
            1: UNK_TOKEN
        }

    def build(self, texts: List[str]) -> None:
        counter = Counter()

        for text in texts:
            tokens = tokenize(text)
            counter.update(text)
    
        most_common = counter.most_common(
            self.max_size - 2
        )

        for idx, (token, _) in enumerate(
            most_common,
            start=2
        ):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def encode(self, text: str) -> List[int]:
        tokens = tokenizer(text)
        return [
            self.token_to_id.get(
                token, 
                self.token_to_id[UNK_TOKEN]
            )
            for token in tokens
        ]

    def decode(self, ids: List[int]) -> str:
        tokens = [
            self.id_to_token.get(
                idx, 
                UNK_TOKEN
            )
            for idx in ids
        ]
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.token_to_id)

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)   
    return tokens

def encode_texts(
    texts: List[str],
    vocabulary: Vocabulary,
    max_length: int        
) -> torch.Tensor:

    encoded_texts = []

    for text in texts:
        ids = vocabulary.encode(text)
        ids = ids[:max_length]
        encoded_texts.append(torch.tensor(ids, dtype=torch.long))

    return encoded_texts

def pad_texts(encoded_texts: List[torch.Tensor]) -> torch.Tensor:
    padded_texts = pad_sequence(
        encoded_texts, 
        batch_first=True, 
        padding_value=0
    )
    return padded_texts

def preprocess_batch(
    texts: List[str],
    vocabulary: Vocabulary,
    max_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    encoded_texts = encode_texts(
        texts=texts, 
        vocabulary=vocabulary, 
        max_length=max_length
    )
    padded_texts = pad_texts(encoded_texts)

    return padded_texts

if __name__ == "__main__":
    texts = [
        "I feel very happy today",
        "I am extremely sad",
        "I am angry about this",
    ]

    vocabulary = Vocabulary(max_size=100)

    vocabulary.build(texts)

    print("Vocabulary size:", len(vocabulary))
    print("\nVocabulary:")
    print(vocabulary.token_to_id)

    encoded = preprocess_batch(
        texts=texts,
        vocabulary=vocabulary,
        max_length=10,
    )

    print("\nEncoded batch:")
    print(encoded)

    print("\nBatch shape:")
    print(encoded.shape)