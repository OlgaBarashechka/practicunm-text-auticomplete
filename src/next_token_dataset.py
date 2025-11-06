from transformers import PreTrainedTokenizerFast, DefaultDataCollator
from torch.utils.data import Dataset, DataLoader

import pandas as pd


# --- Класс датасета ---
class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=20):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.examples = []

        for text in texts:
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding=False)
            input_ids = encodings['input_ids']
            if len(input_ids) > 1:  
                self.examples.append(input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        input_ids_x = input_ids[:-1]
        labels = input_ids[1:]
        return {
            'input_ids': input_ids_x,
            'labels': labels
        }


