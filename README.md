# Emotion-Aware Mental Health Risk Screening (NLP)

This project presents an NLP-based system designed to identify potential mental health risks from textual data using emotion-aware transformer models.

---

## Overview

The system analyzes user-generated text and detects fine-grained emotions using a transformer-based model.
Unlike traditional sentiment analysis, it incorporates contextual understanding and emotion intensity to provide structured psychological risk levels.

---

## Key Features

* Transformer-based emotion detection (RoBERTa – GoEmotions)
* Context-aware processing (negation, intensity)
* Hybrid emotion aggregation
* Risk classification into meaningful categories
* End-to-end NLP pipeline with real-time prediction

---

## Methodology

1. Text preprocessing and tokenization
2. Emotion detection using transformer model
3. Emotion vector generation
4. Contextual enhancement
5. Aggregation of emotional signals
6. Risk scoring and classification

---

## Results

The proposed model achieves higher accuracy compared to traditional approaches:

* TF-IDF: 78%
* LSTM: 83%
* Proposed Model: ~87%

---

## Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Gradio

---

## Live Demo

https://huggingface.co/spaces/Riishab12/emotion-detector-using-nlp

---

## Disclaimer

This project is intended for research and early screening purposes only.
It is not a substitute for professional medical diagnosis.

---

## Author

Riishab Trehan
B.Tech AIML Undergraduate
