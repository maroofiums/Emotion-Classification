from src.config import RANDOM_SEED, DATASET_NAME
from datasets import load_dataset, DatasetDict


def load_emotion_dataset() -> DatasetDict:
    """
    Load the emotion dataset from Hugging Face.

    Returns:
        DatasetDict: A dictionary containing the train, validation, and test datasets.
    """
    dataset = load_dataset(DATASET_NAME)
    return dataset

def get_dataset_info(dataset: DatasetDict) -> None:
    """
    Print information about the dataset.

    Args:
        dataset (DatasetDict): The dataset to get information from.
    """
    print("\nDataset Info:")
    print("-"*40)

    for split in dataset:
        print(f"{split:12}: {len(dataset[split])} samples")

    print("-"*40)

    print("\nFeatures:")

    print("-"*40)

    print(dataset["train"].features)

def show_samples(dataset: DatasetDict, num_samples: int = 5) -> None:
    """
    Show a few samples from the dataset.

    Args:
        dataset (DatasetDict): The dataset to show samples from.
        num_samples (int): The number of samples to show. Default is 5.
    """
    samples = dataset["train"].shuffle(seed=RANDOM_SEED).select(range(num_samples))

    print("\nSample Data:")
    print("-"*40)

    for i, sample in enumerate(samples):
        print(f"Sample {i+1}:")
        print(f"Text: {sample['text']}")
        print(f"Label: {sample['label']}")
        print("-"*40)

if __name__ == "__main__":
    # Load the dataset
    dataset = load_emotion_dataset()

    # Get dataset information
    get_dataset_info(dataset) 

    # Show sample data
    show_samples(dataset, num_samples=5)
    
