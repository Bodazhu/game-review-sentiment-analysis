# Game Review Sentiment Analysis

**NLP-Based Sentiment Classification for Player Feedback Analysis**

An end-to-end machine learning project that classifies Amazon video game reviews by sentiment and explores how automated text classification could help product and customer-support teams identify dissatisfied customers more efficiently.

## Project Overview

Online game reviews contain valuable feedback about player experiences, but reviewing large volumes of free-form text manually is time-consuming. This project investigates whether conventional NLP feature representations and supervised classifiers can distinguish lower-rated reviews from positive reviews.

**Objective:** Predict sentiment from review titles and text, then compare models using classification metrics and discuss the operational trade-off between missed negative reviews and unnecessary review flags.

> **Scope:** This is an offline classification study, not a deployed customer-support system or a model that predicts observed customer churn.

## Dataset

- **Source:** [Amazon Reviews 2023 — Video Games](https://amazon-reviews-2023.github.io/) (UCSD McAuley Lab)
- **Input fields:** Review `title`, `text`, and `rating`
- **Records loaded:** 150,000 (first records from the source JSONL file, capped for Colab runtime)
- **Records after text cleaning:** 149,785
- **Balanced modeling sample:** 66,588 reviews (33,294 per class)
- **Train/test split:** 80/20 stratified split (53,270 training; 13,318 testing)

**Label definition in the notebook:** Ratings of **4–5 stars** are labeled *Positive* (`1`); ratings **below 4 stars**, including 3-star reviews, are labeled *Negative* (`0`). These are **rating-derived proxy labels**, not independently annotated sentiment judgments.

To manage class imbalance in this experiment, the notebook randomly undersamples the majority class before splitting the data.

## Methodology

### 1. Exploratory Data Analysis

- Examined rating and class distributions
- Compared review text lengths by class
- Visualized frequent terms with word clouds

### 2. Text Preprocessing

- Combined review titles and bodies
- Removed empty reviews
- Lowercased text and removed non-alphabetic characters
- Removed English stopwords and applied Porter stemming

### 3. Feature Extraction

Compared three sparse text representations, each limited to 20,000 features:

| Representation | Configuration |
| --- | --- |
| Bag of Words | Unigram counts |
| TF-IDF | Unigram TF-IDF |
| TF-IDF N-Grams | Unigrams, bigrams, and trigrams |

### 4. Model Benchmarking

Trained and evaluated nine combinations of feature representations and classifiers:

- Logistic Regression
- Linear Support Vector Machine (`LinearSVC`)
- Random Forest

Additionally, used **3-fold GridSearchCV** to tune Random Forest hyperparameters (`n_estimators`, `max_depth`, and `min_samples_split`) with weighted F1 as the selection metric.

## Results

**Top-performing feature representation for each algorithm, as recorded in the notebook:**

| Classifier | Representation | Accuracy | Weighted F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| **Logistic Regression** | **TF-IDF N-Grams** | **0.8856** | **0.8856** | **0.9538** |
| Linear SVM | TF-IDF N-Grams | 0.8776 | 0.8776 | 0.9475 |
| Random Forest | TF-IDF N-Grams | 0.8200 | 0.8200 | 0.9068 |

**Best observed benchmark:** Logistic Regression with TF-IDF N-Grams achieved **88.56% accuracy**, **0.8856 weighted F1**, and **0.9538 ROC-AUC** on the balanced test set.

For this model, the saved confusion matrix shows **5,959 of 6,659 negative-labeled reviews correctly identified** (approximately **89.5% negative-class recall**), with **700 negative reviews missed**.

The separately tuned Random Forest achieved a best cross-validation weighted F1 of **0.8391**; this tuning exercise does not change the top benchmark above.

## Business Interpretation

A review triage workflow would care about more than overall accuracy:

- **Missed negative reviews:** Dissatisfied customers whose reviews are not flagged for follow-up.
- **Unnecessary flags:** Positive reviews sent for manual review, consuming team capacity.
- **Negative-class recall:** A relevant metric when missing critical complaints is especially costly.

The results suggest that conventional linear classifiers with TF-IDF N-Grams are a strong baseline for detecting rating-derived negative feedback in this sample. They do **not** demonstrate actual reductions in churn, faster support response, or production deployment.

## Limitations

- **Proxy labels:** Star ratings may not perfectly reflect the sentiment expressed in review text; 3-star reviews are grouped with negative reviews.
- **Sampling:** The analysis uses the first 150,000 source records, not a random sample of the full corpus.
- **Balanced test data:** Majority-class undersampling produces a 50/50 evaluation set, which may not represent real-world review prevalence or precision under deployment conditions.
- **Model selection:** Multiple models were compared using the same holdout set; the reported benchmark is exploratory rather than a fully independent final estimate.
- **Cross-validation preprocessing:** Text vectorization was fitted on the training set before Random Forest cross-validation rather than independently within each CV fold.
- **Business outcomes:** Customer support routing and downstream business impact were discussed conceptually, not implemented or measured.

## Tech Stack

**Python**, pandas, NumPy, NLTK, scikit-learn, Matplotlib, Seaborn, WordCloud, Google Colab

## Repository Contents

- [`Sentiment_Analysis_VideoGames.ipynb`](Sentiment_Analysis_VideoGames.ipynb) — full notebook with preprocessing, model comparisons, tuning, evaluation, and visualizations.

## How to Run

1. Open the notebook in **Google Colab**.
2. Run cells from top to bottom. The notebook installs required Python packages and downloads the source dataset from UCSD McAuley Lab.
3. Allow time and storage for the compressed source dataset (approximately **776.5 MB** in the saved run) and model training.

The notebook processes up to `MAX_RECORDS = 150_000` reviews by default to keep the experiment manageable.

## Data Attribution

Amazon Reviews 2023 dataset, UCSD McAuley Lab. See the [dataset project page](https://amazon-reviews-2023.github.io/) for documentation and citation information.
