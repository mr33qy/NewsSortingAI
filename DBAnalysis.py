import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import re

def clean_text(text):
    # Clean text by converting to lower case, removing digits, punctuation, and extra spaces.
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
#Stemming
def stem_text(text):
    # Initialize the PorterStemmer
    stemmer = PorterStemmer()
    # Tokenize the text to apply stemming
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Join the stemmed tokens back into a string
    return ' '.join(stemmed_tokens)
def load_and_visualize_data():
    # Load data, remove unnecessary columns, handle missing values, and visualize category distribution.
    data = pd.read_csv('News_Categories.csv')
    data.drop(columns=['Unnamed: 0', 'link'], inplace=True)
    data.dropna(subset=['headline', 'short_description'], inplace=True)

    total_articles = data.shape[0]
    category_counts = data['category'].value_counts()
    unique_categories = data['category'].nunique()
    average_articles_per_category = int(total_articles / unique_categories)

    print(category_counts)
    print(f"Average number of articles per category: {average_articles_per_category}")

    # Visualization of article counts per category
    plt.figure(figsize=(12, 8))
    category_counts.plot(kind='bar')
    plt.title('Number of Articles per Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=90)
    plt.show()

    user_choice = input("Do you want to balance data? (yes/no): ")
    return user_choice.strip().lower() == 'yes', data, average_articles_per_category

def clean_and_process_data(data):
    # Cleaning text data in headlines and descriptions
    data['clean_headline'] = data['headline'].apply(clean_text).apply(stem_text)
    data['clean_description'] = data['short_description'].apply(clean_text).apply(stem_text)

    # Combining cleaned headline and description for richer model features
    data['combined_text'] = data['clean_headline'] + ' ' + data['clean_description']

    # Engineering a new feature: the length of the combined text
    data['text_length'] = data['combined_text'].apply(len)

    # Removing stopwords and vectorizing text for model input
    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    features = vectorizer.fit_transform(data['combined_text'])
    labels = data['category']
    
    return features, labels
#merging Process
def process_merge(data):
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
    
def calculate_new_average(data):
    # Calculate the new average number of articles per category after merging
    total_articles = data.shape[0]
    unique_categories = data['category'].nunique()
    return int(total_articles / unique_categories)

def process_data(data, new_average_articles_per_category):
    # Balance the dataset 
    balanced_data = pd.DataFrame()
    for category in data['category'].unique():
        category_slice = data[data['category'] == category]
        if len(category_slice) > new_average_articles_per_category:
            # Reduce categories with more articles than average
            downsampled_slice = category_slice.sample(n=new_average_articles_per_category,random_state=42)
        elif len(category_slice) < new_average_articles_per_category:
            # Increase categories with fewer articles than average
            downsampled_slice = category_slice.sample(n=new_average_articles_per_category, replace=True, random_state=42)
        else:
            downsampled_slice = category_slice

        balanced_data = pd.concat([balanced_data, downsampled_slice], ignore_index=True)

    # Shuffle the balanced dataset
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    print(balanced_data['category'].value_counts())
    return balanced_data
#Apllying RF-IDF Vectorization
def apply_vectorization(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_features = tfidf_vectorizer.fit_transform(data['processed_text'])

    # Apply Count Vectorization
    count_vectorizer = CountVectorizer(stop_words='english')
    count_features = count_vectorizer.fit_transform(data['processed_text'])


    return tfidf_features, count_features
def alternative_action():
    # Action to take if the user decides not to proceed with data balancing.
    print("Exiting the program. Thank you!")
#Split data into training, validation, and testing sets.
def split_data(features, labels, test_size=0.3, val_size=0.5):
    # Split data into initial train and temporary test sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=test_size, random_state=42)
    # Split the temporary test set into actual validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'Logistic Regression': LogisticRegression()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, predictions),
            'Precision': precision_score(y_test, predictions, average='macro'),
            'Recall': recall_score(y_test, predictions, average='macro'),
            'F1 Score': f1_score(y_test, predictions, average='macro')
        }
    return results
def main():
    # Main function to orchestrate the data loading, processing, and optional steps based on user input.
    user_wants_to_proceed, data,new_average_articles_per_category = load_and_visualize_data()
    #cleaning and merging data
    process_merge(data)
    new_average_articles_per_category = calculate_new_average(data)
    
    
    if user_wants_to_proceed:
        if new_average_articles_per_category is None:
            # Proceed without balancing
            print("Proceeding without balancing the data.")
            features, labels = clean_and_process_data(data)
        else:
            # Balance the data
            data = process_data(data, new_average_articles_per_category)
            features, labels = clean_and_process_data(data)
        
        
    # Clean and process data after balancing or deciding not to balance.
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels)

    print(f"Training Set Size: {X_train.shape[0]}")
    print(f"Validation Set Size: {X_val.shape[0]}")
    print(f"Testing Set Size: {X_test.shape[0]}")

if __name__ == "__main__":
    main()
