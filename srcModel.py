from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset

class PashtuDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_model_and_tokenizer(config):
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])
    model = BertForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels']
    )
    return model, tokenizer