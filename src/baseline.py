import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from datasets import load_dataset


class DataLoader:
    def __init__(self):
        self.dataset = load_dataset(
            "csv",
            data_files={
                "train": "./datasets/train_set.csv",
                "dev": "./datasets/dev_set.csv",
            },
        )
        batch_size = 8
        num_of_batches = len(self.dataset) / batch_size


def main():
    pass


if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     dev = torch.device("cuda:0")
    #     print("Running on the GPU")
    # else:
    #     dev = torch.device("cpu")
    #     print("Running on the CPU")
    main()
