import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "SamLowe/roberta-base-go_emotions"

labels = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    return text

# =========================
# EMOTION GROUPS
# =========================
NEGATIVE = ["sadness", "anger", "fear", "remorse", "disappointment", "nervousness"]
POSITIVE = ["joy", "love", "amusement", "gratitude", "optimism"]

NEGATION = ["not", "never", "no", "nothing", "dont", "don't"]

# =========================
# RISK WEIGHTS (IEEE FIX)
# =========================
RISK_WEIGHTS = {
    "sadness": 1.3,
    "anger": 1.2,
    "fear": 1.4,
    "remorse": 1.5,
    "disappointment": 1.1,
    "nervousness": 1.2,
    "neutral": 0.2
}

# =========================
# ANALYZE
# =========================
def analyze(text):
    text = clean_text(text)
    sentences = re.split(r'[.!?\n]+', text)

    all_probs = []

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue

        inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

        # NEGATION HANDLING
        if any(word in sent.lower() for word in NEGATION):
            for emo in POSITIVE:
                probs[labels.index(emo)] *= 0.4
            for emo in NEGATIVE:
                probs[labels.index(emo)] *= 1.5

        # INTENSITY
        if any(word in sent.lower() for word in ["very", "extremely", "really"]):
            probs *= 1.3

        # RECENCY WEIGHT
        weight = (i + 1) / len(sentences)
        probs = probs * weight

        all_probs.append(probs)

    if len(all_probs) == 0:
        return "neutral", [], np.zeros(len(labels))

    # HYBRID AGGREGATION (IEEE FIX)
    mean_scores = np.mean(all_probs, axis=0)
    max_scores = np.max(all_probs, axis=0)

    scores = (0.6 * mean_scores) + (0.4 * max_scores)
    scores = scores / (np.sum(scores) + 1e-9)

    primary = labels[np.argmax(scores)]

    top_idx = np.argsort(scores)[::-1][:5]
    top = [(labels[i], float(scores[i])) for i in top_idx]

    return primary, top, scores

# =========================
# RISK ENGINE (FINAL FIX)
# =========================
def compute_risk(scores):
    risk_score = 0

    for emo, weight in RISK_WEIGHTS.items():
        if emo in labels:
            risk_score += scores[labels.index(emo)] * weight

    pos = sum(scores[labels.index(e)] for e in POSITIVE)

    # FINAL FORMULA (IEEE STYLE)
    risk_score = risk_score - (pos * 0.4)

    return risk_score

def classify(risk):
    if risk < 0.3:
        return "No Significant Risk 🟢", "Stable emotional condition detected."
    elif risk < 0.8:
        return "Emerging Emotional Strain 🟡", "Mild psychological stress indicators present."
    elif risk < 1.5:
        return "Elevated Psychological Risk 🟠", "Consistent distress signals observed."
    else:
        return "Severe Psychological Distress 🔴", "Critical emotional state detected."

# =========================
# TREND
# =========================
def trend(posts):
    if len(posts) == 0:
        return "No data"

    neg_count = sum(1 for _, emo in posts if emo in NEGATIVE)
    ratio = neg_count / len(posts)

    if ratio > 0.7:
        return "Strong negative behavioral trend"
    elif ratio > 0.4:
        return "Moderate negative pattern"
    return "No strong negative trend"

# =========================
# MAIN
# =========================
def run_analysis(text):
    if not text.strip():
        return "Enter text", ""

    posts = [p.strip() for p in text.split("\n") if p.strip()]

    if len(posts) == 0:
        return "No input", ""

    all_scores = []
    post_results = []

    for p in posts:
        emo, _, scores = analyze(p)
        all_scores.append(scores)
        post_results.append((p, emo))

    # FINAL AGGREGATION
    avg = np.mean(all_scores, axis=0)

    primary = labels[np.argmax(avg)]

    # RISK
    risk_score = compute_risk(avg)
    level, msg = classify(risk_score)

    # TREND
    t = trend(post_results)

    # BREAKDOWN
    top_idx = np.argsort(avg)[::-1][:5]
    breakdown = "\n".join(
        [f"{labels[i].upper()} → {avg[i]*100:.2f}%" for i in top_idx]
    )

    # POSTS
    post_summary = "\n".join(
        [f"- {p[:60]} → {emo}" for p, emo in post_results]
    )

    report = f"""
{level}
{msg}

Trend Analysis: {t}

--- Post-wise Signals ---
{post_summary}

--- Emotion Breakdown ---
{breakdown}
"""

    return primary.upper(), report

# =========================
# UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("## 🧠 Mental Health Signal Analyzer")

    input_text = gr.Textbox(lines=8, label="Input")

    btn = gr.Button("Analyze")

    primary = gr.Textbox(label="Primary Emotion")
    report = gr.Textbox(label="Analysis Report", lines=18)

    btn.click(run_analysis, inputs=input_text, outputs=[primary, report])

demo.launch()