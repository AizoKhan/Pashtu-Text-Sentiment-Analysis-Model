import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_label_distribution(df, label_col='label', save_path=None):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=label_col, order=df[label_col].value_counts().index)
    plt.title("Label Distribution")
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()