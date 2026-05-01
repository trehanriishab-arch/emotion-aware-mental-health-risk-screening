from flask import Flask, render_template_string, request, jsonify
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# =========================
# APP INIT
# =========================
app = Flask(__name__)

# ✅ FIXED FOR DEPLOYMENT
MODEL_PATH = "SamLowe/roberta-base-go_emotions"

labels = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

# =========================
# LOAD MODEL
# =========================
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
# ANALYSIS FUNCTION (UNCHANGED)
# =========================
def analyze(text):
    text = clean_text(text)
    text_lower = text.lower()

    has_suppression = any(w in text_lower for w in ["don't react","don’t react","don't show","don’t show","internally","building up"])
    has_numbness = any(w in text_lower for w in ["nothing feels","empty","numb","distant","muted","just existing"])
    has_low_intensity = any(w in text_lower for w in ["not really","not much","not intense","just a bit","nothing serious"])
    has_conflict = any(w in text_lower for w in ["but at the same time","logically","i know but","something feels missing","hard to explain"])

    sentences = re.split(r'(?<=[.!?]) +', text)

    emotion_scores = np.zeros(len(labels))
    sentence_results = []

    for sent in sentences:
        if not sent.strip():
            continue

        inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

        if has_low_intensity:
            probs = probs * 0.75

        confidence = np.max(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        weight = (0.6 * confidence) + (0.4 * (1 - entropy))

        emotion_scores += probs * weight

        idx = np.argmax(probs)
        sentence_results.append((sent, labels[idx]))

    emotion_frequency = {}
    for _, emo in sentence_results:
        emotion_frequency[emo] = emotion_frequency.get(emo, 0) + 1

    for emo, freq in emotion_frequency.items():
        idx = labels.index(emo)
        emotion_scores[idx] += freq * 0.25

    for weak in ["approval","admiration","curiosity","optimism"]:
        emotion_scores[labels.index(weak)] *= 0.6

    primary_idx = np.argmax(emotion_scores)
    primary = labels[primary_idx]
    primary_score = emotion_scores[primary_idx]

    SECONDARY_MAP = {
        "nervousness": "stress",
        "anger": "frustration",
        "annoyance": "frustration",
        "disapproval": "frustration",
        "disappointment": "dissatisfaction",
        "sadness": "low mood",
        "confusion": "confusion",
        "realization": "overthinking"
    }

    LOW_PRIORITY = ["approval", "curiosity", "optimism", "admiration"]

    top_indices = np.argsort(emotion_scores)[::-1]
    secondary = []

    for i in top_indices:
        if labels[i] == primary:
            continue

        label_name = labels[i]

        if label_name in LOW_PRIORITY or label_name == "neutral":
            continue

        if emotion_scores[i] >= 0.2 * primary_score:
            mapped = SECONDARY_MAP.get(label_name, label_name)
            if mapped not in secondary:
                secondary.append(mapped)

        if len(secondary) == 3:
            break

    if primary == "neutral":
        for i in top_indices:
            if labels[i] != "neutral":
                primary = labels[i]
                break

    def boost(emotion, value):
        emotion_scores[labels.index(emotion)] += value

    if has_suppression:
        boost("annoyance", 0.6)
        boost("anger", 0.4)
        boost("nervousness", 1.0)

    if has_numbness:
        boost("sadness", 2.0)
        boost("disappointment", 0.8)

    if has_conflict:
        boost("confusion", 1.2)
        boost("realization", 1.0)

    if "thinking" in text_lower or "overthinking" in text_lower:
        boost("nervousness", 1.2)
        boost("realization", 0.5)

    if has_numbness:
        emotion_scores[labels.index("confusion")] *= 0.7

    primary = labels[np.argmax(emotion_scores)]

    DISPLAY_MAP = {
        "nervousness": "STRESS",
        "anger": "FRUSTRATION",
        "disappointment": "DISAPPOINTMENT"
    }

    display_primary = DISPLAY_MAP.get(primary, primary).upper()

    if primary == "nervousness":
        interpretation = "The speaker is experiencing sustained stress and mental pressure."
    elif primary in ["anger","annoyance"]:
        interpretation = "The speaker shows signs of internal frustration building over time."
    elif primary == "sadness":
        interpretation = (
            "The speaker is experiencing emotional numbness or detachment. "
            "There is reduced emotional responsiveness and disconnection."
        )
    elif primary == "confusion":
        interpretation = "The speaker is experiencing internal conflict and uncertainty."
    elif primary == "disappointment":
        interpretation = "The speaker feels dissatisfaction despite effort."
    else:
        interpretation = "The emotional tone is layered."

    key_points = []
    details = []

    for sent, emo in sentence_results:
        s = sent.strip()

        if emo in ["joy","love","amusement"]:
            key_points.append("Positive moments are present.")
            details.append(f"Positive: \"{s}\"")
        elif emo in ["anger","annoyance","disapproval"]:
            key_points.append("Frustration detected.")
            details.append(f"Frustration: \"{s}\"")
        elif emo in ["sadness","disappointment","grief"]:
            key_points.append("Low emotional state.")
            details.append(f"Low: \"{s}\"")
        elif emo in ["fear","nervousness"]:
            key_points.append("Stress detected.")
            details.append(f"Stress: \"{s}\"")
        elif emo == "confusion":
            key_points.append("Confusion present.")
            details.append(f"Confusion: \"{s}\"")

    key_points = list(set(key_points))
    secondary_text = ", ".join(secondary[:2]) if secondary else ""

    summary_parts = []

    if primary == "nervousness":
        summary_parts.append("The speaker is under sustained stress and pressure.")
    elif primary == "sadness":
        summary_parts.append("The speaker is experiencing emotional numbness and reduced engagement.")
    elif primary in ["anger","annoyance"]:
        summary_parts.append("There is ongoing internal frustration building over time.")
    elif primary == "disappointment":
        summary_parts.append("The speaker feels dissatisfaction despite effort.")
    else:
        summary_parts.append("The emotional state is layered.")

    if secondary_text:
        summary_parts.append(f"Secondary signals such as {secondary_text} are present.")

    summary_parts.append("There is internal conflict in emotional processing.")

    summary = " ".join(summary_parts)

    return {
        "primary": display_primary,
        "secondary": secondary,
        "confidence": "Weighted multi-sentence analysis",
        "interpretation": interpretation,
        "meaning": {
            "summary": summary,
            "points": key_points,
            "details": details
        }
    }

# =========================
# HTML (FULL ORIGINAL)
# =========================
HTML = """<!DOCTYPE html>
<html>
<head>
<title>Emotion Detector</title>
<style>
body { background:#0e1117; color:white; font-family:Arial; padding:40px;}
textarea { width:100%; height:160px; background:#1c1f26; color:white; border:none; padding:10px; border-radius:8px;}
button { padding:10px 20px; margin-top:10px; border:none; background:#00ff88; color:black; border-radius:6px; cursor:pointer;}
.card { background:#1c1f26; padding:20px; margin-top:20px; border-radius:10px;}
h2 { color:#00ff88; }
h3 { margin-top:15px; color:#cccccc; }
li { margin-bottom:6px; }
</style>
</head>
<body>

<h1>🧠 Emotion Detection System</h1>

<form method="POST">
<textarea name="text">{{ text | default('') }}</textarea>
<button type="submit">Analyze Emotion</button>
</form>

{% if result %}

<div class="card">
<h2>🧠 Overall Emotion</h2>
<p><b>{{ result.primary }}</b><br>
Secondary: {{ result.secondary }}<br>
Confidence: {{ result.confidence }}</p>
</div>

<div class="card">
<h2>🧾 Interpretation</h2>
<p>{{ result.interpretation }}</p>
</div>

<div class="card">
<h2>📖 Full Meaning</h2>

<h3>Summary</h3>
<p>{{ result.meaning.summary }}</p>

<h3>Key Signals</h3>
<ul>
{% for p in result.meaning.points %}
<li>{{ p }}</li>
{% endfor %}
</ul>

<h3>Detailed Breakdown</h3>
<ul>
{% for d in result.meaning.details %}
<li>{{ d }}</li>
{% endfor %}
</ul>

</div>

{% endif %}

</body>
</html>
"""

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        if text.strip():
            result = analyze(text)
    return render_template_string(HTML, result=result, text=text)

@app.route("/analyze", methods=["POST"])
def analyze_api():
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"})
    return jsonify(analyze(text))

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run()