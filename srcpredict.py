import torch
from src.data_loader import load_config
from src.model import get_model_and_tokenizer
import os

def predict_text(text):
    config = load_config()
    model, tokenizer = get_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = f"{config['training']['output_dir']}/best_model.pt"
    if not os.path.exists(model_path):
        print("Model not found! Train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config['model']['max_length'],
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
    pred_id = torch.argmax(logits, dim=1).item()

    reverse_map = {v: k for k, v in {"Positive": 0, "Negative": 1, "Neutral": 2, "Ambiguous": 3, "Mixed": 4}.items()}
    return reverse_map[pred_id]

if __name__ == "__main__":
    print("Pashtu Sentiment Analyzer")
    print("Type 'quit' to exit.\n")
    while True:
        txt = input("Enter Pashto text: ").strip()
        if txt.lower() in {'quit', 'exit', 'q'}:
            break
        if txt:
            sentiment = predict_text(txt)
            print(f"→ {sentiment}\n")