import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from src.data_loader import load_data, load_config
from src.model import get_model_and_tokenizer, PashtuDataset

def evaluate():
    config = load_config()
    _, test_df, label_map = load_data()
    model, tokenizer = get_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = f"{config['training']['output_dir']}/best_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found! Run train.py first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_dataset = PashtuDataset(test_df['Text'].tolist(), test_df['label_id'].tolist(), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='weighted')
    report = classification_report(trues, preds, target_names=label_map.keys(), digits=4)

    os.makedirs("outputs/results", exist_ok=True)
    with open("outputs/results/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1: {f1:.4f}\n\n")
        f.write(report)

    print(f"\nAccuracy: {acc:.4f} | F1: {f1:.4f}")
    print("Full report saved to outputs/results/evaluation_report.txt")

if __name__ == "__main__":
    evaluate()