# Airline Review Sentiment Analysis: Classical ML vs Transformers

An end-to-end NLP project that classifies 23,000+ airline passenger reviews as *Recommended* or *Not Recommended*. It compares a classical TF-IDF baseline against a fine-tuned DistilBERT transformer, adds semantic search with sentence embeddings, and audits the models for responsible-AI risks.

> Completed as a project for the Natural Language Processing module of the M.Sc. Artificial Intelligence at Berlin School of Business and Innovation (BSBI).

---

## Business problem

Airlines receive thousands of free-text reviews. Reading them manually is slow, and star ratings alone do not explain *why* passengers are unhappy.

**Goal:** automatically classify review sentiment, surface the service themes customers talk about, and find out whether an expensive transformer model is actually worth it compared with a simple, fast baseline.

---

## Dataset

- **Source:** [Airline Reviews dataset on Kaggle](https://www.kaggle.com/datasets/juhibhojani/airline-reviews) (not included in this repository; the notebook downloads it with `kagglehub`)
- **Size after cleaning:** 23,046 reviews across 497 airlines (125 duplicate reviews removed)
- **Target:** the `Recommended` column (yes / no)
- **Class balance:** 66.2% Not Recommended, 33.8% Recommended, so macro F1 is used alongside accuracy

---

## Approach

| Step | What was done |
|------|---------------|
| 1. Cleaning and EDA | Removed duplicates and invalid labels, analyzed class balance, review length and rating distributions |
| 2. Text preprocessing | Built two pipelines: NLTK (tokenization, stop-word removal, lemmatization) and spaCy (lemmatization plus named entity recognition) |
| 3. Named entity recognition | Extracted 132,217 entities (locations, organizations, dates, money, facilities) to understand what reviews mention |
| 4. Classical baseline | TF-IDF (unigrams + bigrams) with Logistic Regression and Linear SVM, using a stratified 80/20 split |
| 5. Semantic embeddings | Encoded reviews with a Sentence Transformer (384 dimensions), built cosine-similarity search, and visualized the embedding space with PCA and t-SNE |
| 6. Transformer fine-tuning | Fine-tuned `distilbert-base-uncased` with Hugging Face Trainer (3 epochs, FP16, best checkpoint selected on validation macro F1) |
| 7. Fair comparison | Retrained the baseline on the exact same 3,200 / 400 train-test split as DistilBERT so both models are compared on identical data |
| 8. Responsible-AI audit | Analyzed class-specific errors, high-confidence mistakes, input truncation and personal-data exposure in review text |

---

## Results

### Classical models (full dataset, 4,610 test reviews)

| Model | Accuracy | Weighted F1 | Macro F1 | Training time |
|-------|----------|-------------|----------|---------------|
| **TF-IDF + Logistic Regression** | **90.3%** | **0.904** | **0.894** | 0.5 s |
| TF-IDF + Linear SVM | 89.9% | 0.899 | 0.888 | 0.3 s |

### Fair comparison (identical 400-review test set)

<!-- Check these numbers against cells 74, 79 and 81 of your final run before committing -->

| Model | Accuracy | Macro F1 | Parameters | Training time |
|-------|----------|----------|------------|---------------|
| TF-IDF + Logistic Regression | 89.5% | 0.885 | 25K | 1.4 s |
| **Fine-tuned DistilBERT** | **92.0%** | **0.911** | 67M | 90 s |

DistilBERT is **+2.5 accuracy points** better, but needs about **66× the training time** and **2,671× the parameters**, plus a GPU.

---

## Key findings

- **The baseline is hard to beat.** A simple TF-IDF model gets about 90% accuracy in under two seconds. The transformer improves on it, but only by a few points at a much higher cost. For a high-volume, low-risk task, the baseline may be the better business choice.
- **Negative reviews are longer.** Not Recommended reviews average 144 words vs 105 for Recommended ones; unhappy customers explain more.
- **Semantic search works.** Queries like *"flight delayed"*, *"rude customer service"* and *"lost baggage"* returned clearly relevant reviews (cosine similarity around 0.6–0.7), useful for grouping complaints by theme without labels.
- **Mixed reviews cause most errors.** Reviews that describe a bad experience handled well (or a good flight with one serious problem) are misclassified by both model types.
- **Confidence is not correctness.** Most DistilBERT errors were made with over 80% confidence, so a confidence threshold alone would not catch them.
- **Long reviews are truncated.** 14.6% of reviews exceed DistilBERT's 256-token limit, so part of their text is never seen by the model.
- **Reviews contain personal data.** NER finds names, places, dates and booking details, so a production system would need redaction and data-minimization rules.

---

## Tech stack

- **Python**, **Pandas**, **NumPy**
- **NLTK**, **spaCy** for preprocessing and NER
- **scikit-learn** for TF-IDF, Logistic Regression, Linear SVM, PCA, t-SNE and evaluation
- **Sentence Transformers** for embeddings and semantic search
- **Hugging Face Transformers**, **Datasets**, **PyTorch** for DistilBERT fine-tuning
- **Matplotlib**, **WordCloud** for visualization
- **Google Colab** (T4 GPU)

---

## How to run

1. Open the notebook in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU** (required for DistilBERT fine-tuning with FP16).
3. Click **Runtime → Run all**. The first cell downloads the dataset from Kaggle; log in if prompted.
4. The spaCy step processes all 23K reviews and can take 5–10 minutes.

Results can differ slightly between runs because transformer fine-tuning is not fully deterministic.

---

## Author

**Keshav Mittal**: M.Sc. Artificial Intelligence student, Berlin
[LinkedIn](https://www.linkedin.com/in/keshavmittal-ai/) · [GitHub](https://github.com/TechwithKeshav)
