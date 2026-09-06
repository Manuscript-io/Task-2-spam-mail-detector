# Spam Mail Detector

A machine learning project that classifies SMS messages as **Spam** or **Ham** using Natural Language Processing (NLP) and machine learning techniques.

## Project Overview

Spam messages are unwanted messages that may contain advertisements, promotional offers, fraudulent content, or misleading information. The goal of this project is to build a machine learning model that can automatically identify whether an SMS message is spam or a legitimate message (ham).

The project compares **Multinomial Naive Bayes** and **Logistic Regression** models and evaluates their performance using standard classification metrics.

## Dataset

The project uses the **SMS Spam Collection Dataset**.

- Total messages: **5,572**
- Original classes:
  - Ham: 4,825
  - Spam: 747
- Duplicate messages were identified and removed.
- Duplicate records removed: **403**
- Final dataset size: **5,169 messages**

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Workflow

The project follows these major steps:

### 1. Data Loading

The SMS Spam Collection dataset is loaded using Pandas and separated into two columns:

- `label` — Spam or Ham
- `message` — SMS text

### 2. Data Exploration

The dataset is explored by checking:

- Dataset shape
- Class distribution
- Duplicate records

### 3. Data Cleaning

Duplicate messages are removed to avoid unnecessary repetition in the dataset.

The message labels are converted into numerical values:

- Ham → `0`
- Spam → `1`

### 4. Text Preprocessing

The SMS messages are cleaned before feature extraction.

The preprocessing includes:

- Converting text to lowercase
- Removing non-alphabetic characters
- Splitting text into words
- Removing English stopwords

### 5. Train-Test Split

The cleaned dataset is divided into training and testing sets.

- Training data: **4,135 messages**
- Testing data: **1,034 messages**

### 6. TF-IDF Feature Extraction

The cleaned messages are converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The vectorizer was configured with:

- Maximum features: **5,000**
- N-gram range: **1 to 2**

This allows the model to use both individual words and pairs of words as features.

### 7. Machine Learning Models

Two machine learning models were trained:

#### Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification problems and works well with word-based features.

#### Logistic Regression

Logistic Regression was used as a second classification model to compare its performance with Naive Bayes.

## Model Performance

The models were evaluated using accuracy, precision, recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Naive Bayes | 96.71% | 100.00% | 74.04% | 85.09% |
| Logistic Regression | 95.55% | 94.74% | 68.70% | 79.65% |

### Result

**Multinomial Naive Bayes performed better than Logistic Regression** on the test dataset.

It achieved:

- **96.71% Accuracy**
- **100.00% Precision**
- **74.04% Recall**
- **85.09% F1 Score**

## Confusion Matrix

The Naive Bayes confusion matrix produced the following results:

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 903 | 0 |
| Spam | 34 | 97 |

This shows that:

- 903 ham messages were correctly classified as ham.
- 97 spam messages were correctly classified as spam.
- 34 spam messages were incorrectly classified as ham.
- No ham messages were incorrectly classified as spam.

## Testing on New Messages

The trained Naive Bayes model was also tested on new SMS messages that were not part of the original dataset.

Example predictions included:

- Prize/reward message → **SPAM**
- Class-related message → **HAM**
- Cash prize message → **SPAM**
- Lecture notes request → **HAM**
- Promotional discount message → **HAM**

This demonstrates how the trained model can be used to classify new messages.

## Model Saving

The trained model and TF-IDF vectorizer were saved using Joblib:

- `spam_classifier.pkl`
- `tfidf_vectorizer.pkl`

These files can be used later to make predictions without retraining the model.
