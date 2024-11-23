import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Loading dataset
try:
    data = pd.read_csv('News_Categories.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path and name.")
#Confirming the successfull load of data    
print(f"Data shape: {data.shape}")
# Removing duplicates 
data.drop_duplicates(inplace=True)

# Preprocessing data
data.dropna(subset=['headline'], inplace=True) 
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)
data['processed_text'] = data['headline'].apply(preprocess_text)

# Merging categories
def merging(data):
    category_replacements = {
      'WORLD NEWS': 'NEWS & POLITICS',
        'THE WORLDPOST': 'NEWS & POLITICS',
        'WORLDPOST': 'NEWS & POLITICS',
        'POLITICS': 'NEWS & POLITICS',
        'ARTS': 'ARTS & ENTERTAINMENT',
        'CULTURE & ARTS': 'ARTS & ENTERTAINMENT',
        'ENTERTAINMENT': 'ARTS & ENTERTAINMENT',
        'COMEDY': 'ARTS & ENTERTAINMENT',
        'COLLEGE': 'EDUCATION',
        'MEDIA': 'ARTS & ENTERTAINMENT',
        'STYLE': 'LIFESTYLE & WELLNESS',
        'STYLE & BEAUTY': 'LIFESTYLE & WELLNESS',
        'HEALTHY LIVING': 'LIFESTYLE & WELLNESS',
        'WELLNESS': 'LIFESTYLE & WELLNESS',
        'HOME & LIVING': 'LIFESTYLE & WELLNESS',
        'GOOD NEWS': 'LIFESTYLE & WELLNESS',
        'GREEN': 'LIFESTYLE & WELLNESS',
        'ENVIRONMENT': 'LIFESTYLE & WELLNESS',
        'IMPACT': 'SOCIAL ISSUES & VOICES',
        'BLACK VOICES': 'SOCIAL ISSUES & VOICES',
        'QUEER VOICES': 'SOCIAL ISSUES & VOICES',
        'WOMEN': 'SOCIAL ISSUES & VOICES',
        'LATINO VOICES': 'SOCIAL ISSUES & VOICES',
        'PARENTS': 'FAMILY & RELATIONSHIPS',
        'PARENTING': 'FAMILY & RELATIONSHIPS',
        'DIVORCE': 'FAMILY & RELATIONSHIPS',
        'WEDDINGS': 'FAMILY & RELATIONSHIPS',
        'FIFTY': 'FAMILY & RELATIONSHIPS',
        'SCIENCE': 'SCIENCE & TECHNOLOGY',
        'TECH': 'SCIENCE & TECHNOLOGY',
        'BUSINESS': 'BUSINESS & FINANCE',
        'MONEY': 'BUSINESS & FINANCE',
        'TASTE': 'FOOD & LIFESTYLE',
        'FOOD & DRINK': 'FOOD & LIFESTYLE'
    }
    data['category'] = data['category'].replace(category_replacements)
    print(data['category'].value_counts())
    return data

data = merging(data)  # Applying the merging function

# Calculate new average after merging for balancing
def new_average_articles_per_category(data):
    total_articles = data.shape[0]
    unique_categories = data['category'].nunique()
    return int(total_articles / unique_categories)

# Balancing the data
def balancing_data(data):
    new_avg = new_average_articles_per_category(data)
    balanced_data = pd.DataFrame()
    for category in data['category'].unique():
        category_slice = data[data['category'] == category]
        if len(category_slice) > new_avg:
            downsampled_slice = category_slice.sample(n=new_avg, random_state=42)
        elif len(category_slice) < new_avg:
            downsampled_slice = category_slice.sample(n=new_avg, replace=True, random_state=42)
        else:
            downsampled_slice = category_slice
        balanced_data = pd.concat([balanced_data, downsampled_slice], ignore_index=True)
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_data

data = balancing_data(data) 

# Split data into training and test sets
# 20% of the data will be used for testing and 80% for training
test_size = 0.2 
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_text'], 
    data['category'], 
    test_size=test_size, 
    random_state=42
)
# Printing characteristics of training data
print(f"Training data shape: {X_train.shape}")
print("Training data category distribution:")
print(y_train.value_counts())

# Printing characteristics of test data
print(f"Test data shape: {X_test.shape}")
print("Test data category distribution:")
print(y_test.value_counts())

# Vectorization
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

hashing_vectorizer = HashingVectorizer(n_features=2**10)
X_train_hashing = hashing_vectorizer.transform(X_train)
X_test_hashing = hashing_vectorizer.transform(X_test)
# Using Cross-Validation for a more accurate estimate
def evaluate_with_cross_validation(model, X, y, cv=5):
    # Perform cross-validation and then print the mean of the results
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validated scores: {scores}")
    print(f"Average cross-validation score: {scores.mean()}")
print("Evaluating Logistic Regression with Cross-Validation")
evaluate_with_cross_validation(LogisticRegression(), X_train_counts, y_train)

# Model training and evaluation
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"Precision: {precision_score(y_test, predictions, average='weighted')}")
    print(f"Recall: {recall_score(y_test, predictions, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()
# Train and evaluate models
print("Logistic Regression with Count Vectorizer")
train_evaluate_model(LogisticRegression(), X_train_counts, X_test_counts, y_train, y_test)

print("Random Forest with TF-IDF Vectorizer")
train_evaluate_model(RandomForestClassifier(), X_train_tfidf, X_test_tfidf, y_train, y_test)

print("SVM with Hashing Vectorizer")
train_evaluate_model(SVC(), X_train_hashing, X_test_hashing, y_train, y_test)
