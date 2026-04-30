import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
import ast

# ================= CONFIG =================
MODEL_PATH = "goemotions_gpu_model"
DATA_PATH = "dev.tsv"
NUM_LABELS = 28
MAX_LEN = 128
BATCH_SIZE = 16
THRESHOLD = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD DATA =================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, sep="\t")

df.columns = ["text", "labels", "id"]
df = df[["text", "labels"]]

def parse_labels(label_value):
    multi_hot = np.zeros(NUM_LABELS)

    # Convert to string (important in case it's numeric)
    label_str = str(label_value)

    # Split on comma
    label_list = label_str.split(",")

    # Convert each label safely
    for label in label_list:
        multi_hot[int(label)] = 1

    return multi_hot

df["labels"] = df["labels"].apply(parse_labels)

# ================= TOKENIZER + MODEL =================
print("Loading tokenizer and model...")

# 🔥 Load original tokenizer (stable, correct)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load your fine-tuned model weights
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ================= DATASET CLASS =================
class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

dataset = EmotionDataset(df["text"].tolist(), df["labels"].tolist())
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ================= THRESHOLD SWEEP =================
thresholds = [0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

all_probs = []
all_true = []

with torch.no_grad():
    total_batches = len(loader)

    for i, batch in enumerate(loader):
        if i % 10 == 0:
            print(f"Processing batch {i+1}/{total_batches}")

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()

        all_probs.append(probs)
        all_true.append(labels)

all_probs = np.vstack(all_probs)
all_true = np.vstack(all_true)

print("\n===== Threshold Sweep Results =====")

for t in thresholds:
    preds = (all_probs >= t).astype(int)

    f1_micro = f1_score(all_true, preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_true, preds, average="macro", zero_division=0)
    precision_micro = precision_score(all_true, preds, average="micro", zero_division=0)
    recall_micro = recall_score(all_true, preds, average="micro", zero_division=0)

    print("--------------------------------------------------")
    print(f"Threshold: {t}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"Precision Micro: {precision_micro:.4f}")
    print(f"Recall Micro: {recall_micro:.4f}")

print("\n===== Per Label F1 Scores (Threshold = 0.05) =====")

best_threshold = 0.05
preds = (all_probs >= best_threshold).astype(int)

for i in range(NUM_LABELS):
    f1 = f1_score(all_true[:, i], preds[:, i], zero_division=0)
    print(f"Label {i}: F1 = {f1:.4f}")