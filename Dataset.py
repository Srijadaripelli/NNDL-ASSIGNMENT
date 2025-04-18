```python
import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word_labels = self.labels[idx]
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        labels = [self.label2id[label] for label in word_labels]
        aligned_labels = [-100] * len(encoding.input_ids)

        word_ids = encoding.word_ids()
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                aligned_labels[i] = -100
            else:
                aligned_labels[i] = labels[word_idx]

        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(aligned_labels)
        return item
