import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import os
from src.data_loader import load_data, load_config
from src.model import get_model_and_tokenizer, PashtuDataset
import yaml

def train():
    config = load_config()
    train_df, test_df, label_map = load_data()
    model, tokenizer = get_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = PashtuDataset(train_df['Text'].tolist(), train_df['label_id'].tolist(), tokenizer, config['model']['max_length'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    best_loss = float('inf')

    os.makedirs(config['training']['output_dir'], exist_ok=True)

    print("Starting training...")
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

        if config['training']['save_best'] and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{config['training']['output_dir']}/best_model.pt")
            print(f"New best model saved!")

if __name__ == "__main__":
    train()