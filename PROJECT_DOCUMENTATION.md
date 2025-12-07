# Fake News Detection Project - Step-by-Step Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Starting Point](#starting-point)
3. [Step 1: Data Cleaning and Preprocessing](#step-1-data-cleaning-and-preprocessing)
4. [Step 2: Feature Extraction](#step-2-feature-extraction)
5. [Step 3: Model Training and Evaluation](#step-3-model-training-and-evaluation)
6. [Step 4: Results and Visualizations](#step-4-results-and-visualizations)
7. [Project Structure](#project-structure)
8. [Key Findings](#key-findings)

---

## Project Overview

This project implements a comprehensive **Fake News Detection System** using machine learning techniques. The goal is to classify news articles as either "Real" (label: 1) or "Fake" (label: 0) using various text feature extraction methods and multiple classification algorithms.

**Main Components:**
- Data preprocessing and cleaning
- Two feature extraction approaches: TF-IDF and Count Vectorizer
- Multiple machine learning models with hyperparameter tuning
- Comprehensive performance evaluation and visualization

---

## Starting Point

### Initial Data
The project started with raw news datasets located in `data/raw/`:

- **Fake.csv**: Contains 23,481 fake news articles
- **True.csv**: Contains 21,417 real news articles

**Total Dataset Size**: 44,898 articles

**Data Structure:**
Each CSV file contains the following columns:
- `title`: Article headline
- `text`: Article body/content
- `subject`: Category/subject of the article
- `date`: Publication date

---

## Step 1: Data Cleaning and Preprocessing

**Notebook**: `data_cleaning.ipynb`

### 1.1 Load Raw Data
- Loaded both `Fake.csv` and `True.csv` files
- Examined dataset shapes and structure

### 1.2 Label Assignment
- Assigned **label 0** to fake news articles
- Assigned **label 1** to real news articles
- Merged both datasets into a single DataFrame

### 1.3 Data Resampling (Shuffling)
- Randomly shuffled the merged dataset to ensure proper distribution
- This prevents any bias from having all fake news first, then all real news

### 1.4 Text Preprocessing Pipeline

The following preprocessing steps were applied to prepare the text data:

#### a. **Merge Title and Text**
- Combined `title` and `text` columns into a single `content` column
- This ensures both headline and body information are used for classification

#### b. **Remove Empty Content**
- Removed rows with null or empty content
- Reset index after removal

#### c. **Tokenization**
- Used NLTK's `word_tokenize` to split text into individual tokens (words)
- Created a new `tokens` column containing lists of tokens

#### d. **Lowercase Conversion**
- Converted all tokens to lowercase for consistency

#### e. **Stop Word Removal**
- Removed common English stop words (e.g., "the", "is", "at", "which")
- Used NLTK's English stopwords list

#### f. **Punctuation and Digit Removal**
- Removed punctuation marks and numeric digits
- Kept only alphabetic characters

#### g. **Lemmatization**
- Applied WordNet lemmatization to convert words to their base forms
- Example: "running" → "run", "better" → "good"

#### h. **HTML Tag and Extra Space Removal**
- Removed HTML tags (if any)
- Normalized whitespace (multiple spaces/tabs/newlines → single space)

#### i. **Reconstruct Clean Text**
- Joined processed tokens back into clean text strings
- Created `clean_text` column for feature extraction

### 1.5 Train-Test Split
- Split the preprocessed data into:
  - **Training set**: 80% (35,918 articles)
  - **Test set**: 20% (8,980 articles)
- Used stratified splitting to maintain class distribution
- Saved both sets to `data/processed/train.csv` and `data/processed/test.csv`

**Output Files:**
- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/cleaned_data.csv`

---

## Step 2: Feature Extraction

Two different feature extraction methods were implemented and compared:

### 2.1 TF-IDF Feature Extraction

**Notebook**: `tf_idf_preprocessing.ipynb` and `tf_idf_preprocessing_ngram.ipynb`

#### a. **N-gram Range Optimization**
- Used **GridSearchCV** with Logistic Regression to find optimal n-gram range
- Tested ranges: `(1,1)`, `(1,2)`, `(1,3)`
- **Best Result**: `(1,3)` - unigrams, bigrams, and trigrams
- **Best F1 Score**: 0.9763

#### b. **TF-IDF Vectorization**
- Applied `TfidfVectorizer` with:
  - `ngram_range=(1,3)`
  - `max_features=10000` (top 10,000 features)
  - `stop_words='english'`
- Created sparse feature matrix from training data
- Saved features and vectorizer to: `tfidf_features_ngram_1_1.pkl`

#### c. **Test Data Transformation**
- Transformed test data using the fitted vectorizer
- Ensured consistent feature space between train and test

### 2.2 Count Vectorizer Feature Extraction

**Notebook**: `tf_idf_preprocessing_ngram.ipynb`

#### a. **N-gram Range Optimization**
- Used **GridSearchCV** with Logistic Regression
- Tested ranges: `(1,1)`, `(1,2)`, `(1,3)`
- **Best Result**: `(1,2)` - unigrams and bigrams
- **Best Accuracy**: 0.9956

#### b. **Count Vectorization**
- Applied `CountVectorizer` with:
  - `ngram_range=(1,2)`
  - `max_features=10000`
  - `stop_words='english'`
- Created feature matrix based on word counts
- Saved features to: `count_features_ngram.pkl`

---

## Step 3: Model Training and Evaluation

**Notebook**: `tf_idf_preprocessing_ngram.ipynb`

### 3.1 Class Balancing with SMOTE

- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance classes
- Generated synthetic samples for minority class
- Created balanced training sets: `X_res` and `y_res`

### 3.2 Models Trained

The following 8 machine learning models were trained and evaluated:

#### 1. **Logistic Regression**
- **Hyperparameter Tuning**: GridSearchCV
- **Parameters Tuned**: `C`, `penalty`, `solver`
- **Best Parameters (TF-IDF)**: `C=100`, `penalty='l2'`, `solver='saga'`
- **Best Parameters (Count Vectorizer)**: `C=0.1`, `penalty='l2'`, `solver='lbfgs'`

#### 2. **Naive Bayes (MultinomialNB)**
- **Hyperparameter Tuning**: GridSearchCV
- **Parameters Tuned**: `alpha`, `fit_prior`
- **Best Parameters**: `alpha=0.01`, `fit_prior=True`

#### 3. **Decision Tree**
- **Hyperparameter Tuning**: GridSearchCV
- **Parameters Tuned**: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Best Parameters**: `criterion='gini'`, `max_depth=20`, `min_samples_leaf=2`, `min_samples_split=5`

#### 4. **Random Forest**
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Parameters Tuned**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- Multiple parameter combinations tested

#### 5. **AdaBoost**
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Parameters Tuned**: `n_estimators`, `learning_rate`, `base_estimator`
- Ensemble method using decision trees as base estimators

#### 6. **Gradient Boosting**
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Parameters Tuned**: `n_estimators`, `learning_rate`, `max_depth`, `subsample`
- Sequential ensemble learning method

#### 7. **XGBoost**
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Parameters Tuned**: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`
- Advanced gradient boosting implementation

#### 8. **Support Vector Machine (SVM)**
- **Hyperparameter Tuning**: GridSearchCV
- **Parameters Tuned**: `C`, `kernel`, `gamma`
- Tested with linear and RBF kernels

### 3.3 Evaluation Metrics

Each model was evaluated using the following metrics on the test set:

1. **Precision**: Proportion of positive identifications that were correct
2. **Recall**: Proportion of actual positives that were identified correctly
3. **F1-Score**: Harmonic mean of precision and recall
4. **Accuracy**: Overall correctness of predictions
5. **ROC-AUC**: Area under the Receiver Operating Characteristic curve

### 3.4 Results Storage

Performance metrics were saved to CSV files:
- `data/result/TFIDF_default_metrics.csv`
- `data/result/TFIDF_tuned_metrics.csv`
- `data/result/count_vectorization_default_metrics.csv`
- `data/result/count_vectorization_tuned_metrics.csv`

---

## Step 4: Results and Visualizations

### 4.1 Performance Comparison Visualizations

Multiple visualization types were generated to compare model performance:

#### a. **Grouped Bar Charts**
- Model-wise comparisons for each metric (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Feature set comparisons (TF-IDF vs Count Vectorizer)
- Setting comparisons (Default vs Tuned)
- All model combinations across different metrics

**Location**: `data/result/graph/comparison/`

**Files:**
- `grouped_bar_chart_allmodels_accuracy_all_combos.png`
- `grouped_bar_chart_allmodels_f1_score_all_combos.png`
- `grouped_bar_chart_allmodels_precision_all_combos.png`
- `grouped_bar_chart_allmodels_recall_all_combos.png`
- `grouped_bar_chart_allmodels_roc_auc_all_combos.png`
- Individual model charts (Logistic Regression, Naive Bayes, Decision Tree, etc.)

#### b. **Radar Charts**
- Multi-dimensional performance visualization for each model
- Shows all metrics simultaneously for easy comparison
- Separate charts for each model across all combinations

#### c. **Heatmaps**
- Correlation and performance heatmaps
- `comparison_heatmap.png`: Overall model comparison
- Individual heatmaps for TF-IDF and Count Vectorizer settings

#### d. **Difference Bar Charts**
- Comparison between different approaches:
  - TF-IDF vs Count Vectorizer (default)
  - TF-IDF vs Count Vectorizer (tuned)
  - TF-IDF tuned vs default
  - Count Vectorizer tuned vs default

#### e. **Line Charts**
- Performance trends across different settings
- Model comparison over different configurations

#### f. **Ranking Charts**
- Model rankings based on different metrics
- `overall_ranking_chart.png`: Comprehensive ranking visualization

### 4.2 Key Performance Results

#### TF-IDF (Tuned) Results:
| Model | Precision | Recall | F1-Score | Accuracy | ROC-AUC |
|-------|-----------|--------|----------|----------|---------|
| Random Forest | 0.9967 | 0.9970 | 0.9968 | 0.9969 | 0.9998 |
| XGBoost | 0.9963 | 0.9970 | 0.9966 | 0.9968 | 0.9996 |
| Logistic Regression | 0.9939 | 0.9932 | 0.9936 | 0.9939 | 0.9995 |
| Decision Tree | 0.9939 | 0.9963 | 0.9951 | 0.9953 | 0.9948 |
| Gradient Boosting | 0.9935 | 0.9967 | 0.9951 | 0.9953 | 0.9973 |
| SVM | 0.9925 | 0.9939 | 0.9932 | 0.9935 | 0.9995 |
| AdaBoost | 0.9886 | 0.9974 | 0.9930 | 0.9933 | 0.9971 |
| Naive Bayes | 0.9482 | 0.9555 | 0.9518 | 0.9540 | 0.9892 |

#### Count Vectorizer (Tuned) Results:
| Model | Precision | Recall | F1-Score | Accuracy | ROC-AUC |
|-------|-----------|--------|----------|----------|---------|
| XGBoost | 0.9972 | 0.9974 | 0.9973 | 0.9974 | 0.9997 |
| Logistic Regression | 0.9958 | 0.9951 | 0.9954 | 0.9957 | 0.9987 |
| Gradient Boosting | 0.9921 | 0.9977 | 0.9949 | 0.9951 | 0.9978 |
| Decision Tree | 0.9942 | 0.9958 | 0.9950 | 0.9952 | 0.9945 |
| SVM | 0.9955 | 0.9944 | 0.9950 | 0.9952 | 0.9991 |
| AdaBoost | 0.9907 | 0.9972 | 0.9939 | 0.9942 | 0.9992 |
| Random Forest | 0.9570 | 0.9756 | 0.9663 | 0.9676 | 0.9933 |
| Naive Bayes | 0.9517 | 0.9602 | 0.9559 | 0.9579 | 0.9824 |

---

## Project Structure

```
Fake New Detection/
│
├── data/
│   ├── raw/
│   │   ├── Fake.csv              # 23,481 fake news articles
│   │   └── True.csv              # 21,417 real news articles
│   │
│   ├── processed/
│   │   ├── cleaned_data.csv      # Fully preprocessed dataset
│   │   ├── train.csv             # Training set (80%)
│   │   └── test.csv               # Test set (20%)
│   │
│   └── result/
│       ├── TFIDF_default_metrics.csv
│       ├── TFIDF_tuned_metrics.csv
│       ├── count_vectorization_default_metrics.csv
│       ├── count_vectorization_tuned_metrics.csv
│       └── graph/
│           ├── comparison/        # Comparison visualizations
│           ├── tfidf/            # TF-IDF specific charts
│           └── count_vectorizer/ # Count Vectorizer charts
│
├── data_cleaning.ipynb            # Step 1: Data preprocessing
├── tf_idf_preprocessing.ipynb     # TF-IDF feature extraction
├── tf_idf_preprocessing_ngram.ipynb  # Main notebook: Feature extraction + Model training
├── tfidf_features_ngram_1_1.pkl  # Saved TF-IDF features
├── count_features_ngram.pkl       # Saved Count Vectorizer features
└── venv/                          # Python virtual environment
```

---

## Key Findings

### 1. **Feature Extraction Comparison**
- **TF-IDF** performed better overall, especially with Random Forest and XGBoost
- **Count Vectorizer** showed excellent performance with XGBoost (highest accuracy: 99.74%)
- Optimal n-gram ranges differ: TF-IDF (1,3) vs Count Vectorizer (1,2)

### 2. **Best Performing Models**
- **XGBoost with Count Vectorizer**: Highest accuracy (99.74%) and F1-score (99.73%)
- **Random Forest with TF-IDF**: Best ROC-AUC (99.98%) and strong overall performance
- **Logistic Regression**: Consistent high performance with both feature extraction methods

### 3. **Model Performance Insights**
- **Ensemble methods** (Random Forest, XGBoost, Gradient Boosting) generally outperformed single models
- **Naive Bayes** showed lower performance compared to other models
- **Hyperparameter tuning** significantly improved model performance across all algorithms

### 4. **Class Balancing Impact**
- **SMOTE** effectively balanced the dataset
- Improved model generalization and reduced bias toward majority class

### 5. **Text Preprocessing Importance**
- Comprehensive preprocessing (tokenization, lemmatization, stop word removal) was crucial
- N-gram features captured important contextual information

---

## Summary

This project successfully implemented a comprehensive fake news detection system through:

1. **Thorough data preprocessing** - Cleaning, tokenization, lemmatization, and normalization
2. **Dual feature extraction** - Comparing TF-IDF and Count Vectorizer approaches
3. **Extensive model evaluation** - Training 8 different algorithms with hyperparameter tuning
4. **Comprehensive visualization** - Multiple chart types for performance analysis
5. **Achieved high accuracy** - Best model achieved 99.74% accuracy

The project demonstrates a complete machine learning pipeline from raw data to final model evaluation, with detailed documentation and visualization of results.

---

