import os
import pandas as pd
import numpy as np
import torch
torch.manual_seed(42)
np.random.seed(42)
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import ast
import torch.nn as nn

# =========================
# CPU PERFORMANCE SETTINGS
# =========================

torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# =========================
# CONFIG
# =========================

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "goemotions_refined_model"
DATA_PATH = "train.tsv"

NUM_LABELS = 28
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 3e-5

device = torch.device("cpu")
print("Using device:", device)
print("CPU threads:", torch.get_num_threads())

# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["text", "emotion_ids", "comment_id"]
)

df = df.dropna(subset=["text", "emotion_ids"])

def to_multihot(label_str):
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    try:
        ids = ast.literal_eval(label_str)
        for i in ids:
            if 0 <= int(i) < NUM_LABELS:
                vec[int(i)] = 1.0
    except:
        pass
    return vec

labels = np.stack(df["emotion_ids"].apply(to_multihot).values).astype(np.float32)

# =========================
# CLASS WEIGHTS
# =========================

label_counts = labels.sum(axis=0)
total_samples = len(labels)

label_counts[label_counts == 0] = 1
pos_weights = np.sqrt((total_samples - label_counts) / label_counts)

# Cap extreme values (VERY IMPORTANT)
pos_weights = np.clip(pos_weights, 1.0, 10.0)

pos_weights = torch.tensor(pos_weights, dtype=torch.float32)

print("Class weights:", pos_weights)

texts = df["text"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.1,
    random_state=42
)

# =========================
# TOKENIZER
# =========================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, truncation=True, padding=False, max_length=MAX_LEN)
val_encodings = tokenizer(val_texts, truncation=True, padding=False, max_length=MAX_LEN)

class GoEmotionsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

train_dataset = GoEmotionsDataset(train_encodings, train_labels)
val_dataset = GoEmotionsDataset(val_encodings, val_labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# MODEL
# =========================

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

model.config.problem_type = "multi_label_classification"
model.to(device)

# =========================
# LOSS FUNCTION (IMPORTANT)
# =========================

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

# =========================
# CUSTOM TRAINER
# =========================

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

# =========================
# METRICS
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.from_numpy(logits))
    preds = (probs > 0.1).int().numpy()

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }

# =========================
# TRAINING ARGS
# =========================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=200,
    save_total_limit=2,
    dataloader_num_workers=0,
    report_to=[]
)

# =========================
# TRAINER
# =========================

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# =========================
# TRAIN
# =========================

print("Starting training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Model saved to:", OUTPUT_DIR)