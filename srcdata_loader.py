import pandas as pd
import yaml
import re
from sklearn.model_selection import train_test_split

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def load_data():
    config = load_config()['data']
    df = pd.read_excel(config['file_path'], sheet_name=config['sheet_name'])
    df = df[[config['text_col'], config['label_col']]].dropna()
    df[config['text_col']] = df[config['text_col']].apply(clean_text)
    
    label_map = {"Positive": 0, "Negative": 1, "Neutral": 2, "Ambiguous": 3, "Mixed": 4}
    df['label_id'] = df[config['label_col']].map(label_map)
    df = df.dropna(subset=['label_id']).reset_index(drop=True)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_id'])
    return train_df, test_df, label_map

if __name__ == "__main__":
    train, test, mapping = load_data()
    print(f"Train: {len(train)}, Test: {len(test)}")