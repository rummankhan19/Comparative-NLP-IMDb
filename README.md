# Comparative NLP — IMDb Sentiment & Emotion Classification

A comparative NLP project built from three IMDb review classification experiments, covering classical machine learning and transformer-based approaches.

## Models

| Approach | Task | Features / Architecture | Key Notes |
|---|---|---|---|
| Logistic Regression | Binary sentiment | TF-IDF, 5,000 features | Text preprocessing, train/test split, classification report, confusion matrix, custom review inference |
| Random Forest | Multi-class emotion | TF-IDF word/bi-gram features, 5,000 features | Heuristic emotion labeling, RandomOverSampler for the training set, classification report and confusion matrix |
| BERT | Binary sentiment | `bert-base-uncased` | Tokenization with max length 128, Hugging Face Trainer, loss curves, confusion matrix, sample predictions |

## Experiments

### 1. Logistic Regression
Traditional linear classification using TF-IDF representations of IMDb reviews. The experiment maps positive reviews to **joy** and negative reviews to **sadness**, then evaluates Logistic Regression on a stratified train/test split.

**Google Colab:**
https://colab.research.google.com/drive/1-_UKro5YYBIlLWbOINkE74AlvbSuxtvE#printMode=true

### 2. Random Forest — Emotion Classification
A multi-class experiment that derives emotion labels from the IMDb review text using sentiment signals, VADER neutral detection, and keyword/synonym heuristics for **Contemplation** and **Concern**, with positive/negative fallbacks to **Joy/Sadness**. The training data is balanced with RandomOverSampler before fitting a Random Forest classifier.

**Google Colab:**
https://colab.research.google.com/drive/1SG7m9AD3jPxOZ0XpeqHxlx07TPH104SE#printMode=true

> Note: The resulting labels are heuristic and highly imbalanced, so accuracy should not be interpreted alone. Macro F1 is a more informative metric for this experiment.

### 3. BERT Sentiment Classification
A transformer-based binary sentiment classifier using `bert-base-uncased`. The notebook demonstrates tokenization, model fine-tuning with Hugging Face Trainer, evaluation metrics, loss visualization, confusion matrix analysis, and example predictions.

The experiment uses a **5,000-review training subset and 2,000-review test subset** from IMDb, with a maximum sequence length of 128 tokens.

**Google Colab:**
https://colab.research.google.com/drive/1WvvjjjsGrYzpQQPhe5uIji1mmDEDKOoh#printMode=true

## Workflow

```text
IMDb Reviews
     |
     +--------------------+----------------------+-------------------+
     |                    |                      |
 TF-IDF              TF-IDF + RF          BERT Tokenizer
     |                    |                      |
 Logistic Regression   Emotion Labels       BERT Fine-tuning
     |                    |                      |
 Sentiment Metrics     Multi-class Metrics  Sentiment Metrics
```

## Technologies

- Python
- scikit-learn
- NLTK
- TF-IDF
- Logistic Regression
- Random Forest
- RandomOverSampler
- VADER
- Hugging Face Transformers
- BERT
- PyTorch
- Google Colab

## Repository Note

The original experiments were developed in Google Colab. This repository serves as a single project hub for the three comparative experiments and their corresponding notebooks/links.

## Learning Objective

The project demonstrates the progression from feature-engineered classical NLP models to a pretrained transformer, while also highlighting the importance of class balance, label quality, and metric selection when comparing NLP systems.