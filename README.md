# Automatic Ticket Classification - NLP Case Study

## Overview
This project implements an automatic ticket classification system using Topic Modeling (NMF) and Supervised Learning to classify financial customer complaints into 5 categories.

## Categories
1. Credit card / Prepaid card
2. Bank account services  
3. Theft/Dispute reporting
4. Mortgages/loans
5. Others

## Dataset
- **File**: complaints-2021-05-14_08_16.json
- **Records**: 78,313+ customer complaints
- **Source**: Financial services customer complaints

## Project Structure
```
├── Automatic_Ticket_Classification.ipynb  # Main Jupyter notebook
├── complaints-2021-05-14_08_16.json       # Dataset
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

## Installation

### 1. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook Automatic_Ticket_Classification.ipynb
```

## Methodology

### Stage 1: Data Loading
- Load JSON data
- Inspect structure and data quality
- Handle missing values

### Stage 2: Text Preprocessing
- Lowercase conversion
- Remove URLs, emails, dates, numbers
- Remove punctuation and stopwords
- Tokenization
- POS tagging and lemmatization

### Stage 3: Exploratory Data Analysis (EDA)
- Word count distribution
- Top frequent words
- Word clouds
- N-gram analysis (bigrams, trigrams)

### Stage 4: Feature Extraction
- TF-IDF vectorization
- Max features: 5000
- N-gram range: (1, 2) - unigrams and bigrams

### Stage 5: Topic Modeling with NMF
- Non-negative Matrix Factorization
- Number of topics: 5
- Elbow method for validation
- Topic-word distribution analysis
- Manual mapping to business categories

### Stage 6: Model Building (Supervised Learning)
- **Models**:
  - Logistic Regression
  - Random Forest Classifier
  - Decision Tree Classifier
  - Naive Bayes (Bonus)
- Train-test split: 80-20
- Class balancing

### Stage 7: Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Classification reports
- Model comparison

### Stage 8: Model Inference
- Test on custom complaints
- Prediction with confidence scores
- Category probability distribution

## Expected Results
- **Topic Modeling**: 5 distinct topics mapped to business categories
- **Classification Accuracy**: >85% on test set
- **Best Model**: Typically Logistic Regression or Random Forest
- **Business Value**: Automated routing saves manual effort

## Key Insights
1. TF-IDF effectively captures document importance
2. NMF creates interpretable topics
3. Supervised models generalize well to new complaints
4. Logistic Regression often performs best due to linear separability

## Next Steps
1. Deploy model as REST API
2. Implement real-time classification pipeline
3. Add sentiment analysis for priority scoring
4. Integrate with CRM/ticketing systems
5. Monitor model drift and retrain periodically

## Author
Praveen Chand  
October 2026

Output cells now include:
✅ Library import confirmation
✅ Data loading results (78,313 records)
✅ Data cleaning statistics (20,930 valid records)
✅ Product distribution table
✅ NLTK resources confirmation
✅ Text preprocessing examples (before/after)
✅ Processing progress
✅ TF-IDF matrix shape and sparsity
✅ NMF topics with top words for each
✅ Category distribution
✅ Train-test split sizes
✅ All 4 models' accuracy scores
✅ Best model comparison table
✅ Inference predictions on test cases
✅ Model save confirmation

## License
Educational/Assignment Purpose
