import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "../MODELS/goemotions_refined_model"
THRESHOLD = 0.1

labels = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().numpy()

    top_k = 3
    top_indices = probs.argsort()[-top_k:][::-1]

    predicted_labels = [
        labels[i] for i in top_indices if probs[i] > THRESHOLD
    ]

    # ✅ FIXED INDENTATION
    if len(predicted_labels) == 0:
        predicted_labels = [labels[top_indices[0]]]

    return predicted_labels, probs


if __name__ == "__main__":
    print("\n=== Emotion Predictor ===")
    print("Type 'exit' to quit\n")

    while True:
        text = input("Enter text: ")

        if text.lower() == "exit":
            break

        preds, probs = predict(text)

        print("\nPredicted Emotions:", preds)

        top_indices = probs.argsort()[-5:][::-1]
        print("\nTop Scores:")
        for i in top_indices:
            print(f"{labels[i]}: {probs[i]:.4f}")

        print("\n" + "-"*50)