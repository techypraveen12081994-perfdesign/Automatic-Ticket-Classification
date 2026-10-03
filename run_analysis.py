#!/usr/bin/env python3
"""
Automatic Ticket Classification - NLP Case Study
Execute this script to run the complete analysis
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AUTOMATIC TICKET CLASSIFICATION - NLP CASE STUDY")
print("="*80)

# Stage 1: Load Data
print("\n[1/8] Loading data...")
with open('complaints-2021-05-14_08_16.json', 'r') as f:
    data = json.load(f)

# Extract _source from each record
records = [item['_source'] for item in data]
df = pd.DataFrame(records)

print(f"✓ Loaded {len(df)} records")
print(f"  Columns: {len(df.columns)}")

# Stage 2: Data Cleaning
print("\n[2/8] Cleaning data...")
df_clean = df.copy()

# Use complaint_what_happened as the text column
df_clean['complaint_text'] = df_clean['complaint_what_happened'].fillna('')

# Remove empty complaints
df_clean = df_clean[df_clean['complaint_text'].str.len() > 10]
df_clean = df_clean.drop_duplicates(subset=['complaint_text'])
df_clean.reset_index(drop=True, inplace=True)

print(f"✓ Cleaned dataset: {len(df_clean)} records")
print(f"  Product distribution:")
for prod, count in df_clean['product'].value_counts().head(5).items():
    print(f"    - {prod}: {count}")

# Stage 3: Text Preprocessing
print("\n[3/8] Preprocessing text...")
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download NLTK data
for resource in ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str) or len(text) < 10:
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    text = re.sub(r'\$\d+(\.\d{2})?', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)

# Process in batches to show progress
batch_size = 5000
df_clean['processed_text'] = ''

for i in range(0, len(df_clean), batch_size):
    end_idx = min(i + batch_size, len(df_clean))
    df_clean.loc[i:end_idx-1, 'processed_text'] = df_clean.loc[i:end_idx-1, 'complaint_text'].apply(preprocess_text)
    print(f"  Processed {end_idx}/{len(df_clean)} records...")

# Remove empty processed text
df_clean = df_clean[df_clean['processed_text'].str.len() > 0]
df_clean.reset_index(drop=True, inplace=True)

print(f"✓ Preprocessing complete: {len(df_clean)} records ready")

# Stage 4: Feature Extraction
print("\n[4/8] Extracting TF-IDF features...")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf_vectorizer.fit_transform(df_clean['processed_text'])
print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"  Features: {tfidf_matrix.shape[1]}")
print(f"  Sparsity: {(1.0 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2%}")

# Stage 5: Topic Modeling with NMF
print("\n[5/8] Building NMF topic model...")
from sklearn.decomposition import NMF

nmf_model = NMF(
    n_components=5,
    random_state=42,
    max_iter=500,
    init='nndsvda'
)

nmf_features = nmf_model.fit_transform(tfidf_matrix)
print(f"✓ NMF model trained")
print(f"  Topics: 5")
print(f"  Reconstruction error: {nmf_model.reconstruction_err_:.2f}")

# Display topics
print("\n  Top words per topic:")
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf_model.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"    Topic {topic_idx}: {' | '.join(top_words)}")

# Assign topics
dominant_topics = np.argmax(nmf_features, axis=1)
df_clean['topic'] = dominant_topics

# Map to categories (based on observed patterns)
topic_labels = {
    0: 'Credit card / Prepaid card',
    1: 'Bank account services',
    2: 'Theft/Dispute reporting',
    3: 'Mortgages/loans',
    4: 'Others'
}

df_clean['category'] = df_clean['topic'].map(topic_labels)

print(f"\n  Category distribution:")
for cat, count in df_clean['category'].value_counts().items():
    print(f"    - {cat}: {count}")

# Stage 6: Supervised Learning
print("\n[6/8] Training supervised learning models...")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

X = df_clean['processed_text']
y = df_clean['topic']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create TF-IDF for supervised learning
tfidf_supervised = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_train_tfidf = tfidf_supervised.fit_transform(X_train)
X_test_tfidf = tfidf_supervised.transform(X_test)

print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, class_weight='balanced', n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42, class_weight='balanced'),
    'Naive Bayes': MultinomialNB(alpha=0.1)
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_tfidf, y_train)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    train_pred = model.predict(X_train_tfidf)
    test_pred = model.predict(X_test_tfidf)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, average='weighted')
    recall = recall_score(y_test, test_pred, average='weighted')
    f1 = f1_score(y_test, test_pred, average='weighted')

    results[name] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"    Train Accuracy: {train_acc:.4f}")
    print(f"    Test Accuracy: {test_acc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall: {recall:.4f}")
    print(f"    F1-Score: {f1:.4f}")

# Stage 7: Model Comparison
print("\n[7/8] Model comparison:")
results_df = pd.DataFrame(results).T
print(results_df.to_string())

best_model_name = results_df['test_accuracy'].idxmax()
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test Accuracy: {results_df.loc[best_model_name, 'test_accuracy']:.4f}")

# Stage 8: Model Inference
print("\n[8/8] Testing model inference...")

def predict_complaint(text, model, vectorizer, labels):
    processed = preprocess_text(text)
    tfidf = vectorizer.transform([processed])
    prediction = model.predict(tfidf)[0]
    probabilities = model.predict_proba(tfidf)[0]

    return {
        'category': labels[prediction],
        'confidence': probabilities[prediction],
        'all_probs': {labels[i]: prob for i, prob in enumerate(probabilities)}
    }

test_complaints = [
    "I lost my credit card and someone made unauthorized charges. I need help disputing these transactions.",
    "My mortgage loan application was rejected without proper explanation.",
    "There was an unauthorized withdrawal from my checking account."
]

print("\n  Test cases:")
for i, complaint in enumerate(test_complaints, 1):
    result = predict_complaint(complaint, best_model, tfidf_supervised, topic_labels)
    print(f"\n  [{i}] {complaint[:80]}...")
    print(f"      → Category: {result['category']}")
    print(f"      → Confidence: {result['confidence']:.2%}")

# Save models
print("\n" + "="*80)
print("SAVING MODELS...")
import pickle

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_supervised, f)

with open('nmf_model.pkl', 'wb') as f:
    pickle.dump(nmf_model, f)

with open('topic_labels.pkl', 'wb') as f:
    pickle.dump(topic_labels, f)

print("✓ Models saved:")
print("  - best_model.pkl")
print("  - tfidf_vectorizer.pkl")
print("  - nmf_model.pkl")
print("  - topic_labels.pkl")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\nSummary:")
print(f"  - Processed {len(df_clean)} complaints")
print(f"  - Identified 5 topic clusters")
print(f"  - Best model: {best_model_name} ({results_df.loc[best_model_name, 'test_accuracy']:.2%} accuracy)")
print(f"  - All outputs generated and models saved")
print("\n" + "="*80)
