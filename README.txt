Pashtu Sentiment Analysis 

---

## Overview

This repository implements a **5-class sentiment classifier** for **Pashto (پښتو)** text using **deep learning**.

### Sentiment Classes
| Label      | ID |
|------------|----|
| Positive   | 0  |
| Negative   | 1  |
| Neutral    | 2  |
| Ambiguous  | 3  |
| Mixed      | 4  |

---

## Dataset

- File: `data/Pashtu Text Sentiment Analysis Dataset.xlsx`
- Sheet: `Sheet1`
- Columns: `Text`, `label`

> **Note:** The dataset contains ~21,800+ labeled Pashto sentences (refugee issues, politics, culture, etc.).

---

## Features

- Modular code (data, model, train, eval, predict)
- Configurable via `config.yaml`
- Reproducible with `requirements.txt`
- Interactive prediction CLI
- Evaluation report saved
- EDA notebook included

---

## Installation

```bash
cd Pashtu-Sentiment-Analysis
pip install -r requirements.txt