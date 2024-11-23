import pandas as pd
import matplotlib.pyplot as plt

def load_and_visualize_data():
    # Load the dataset
    data = pd.read_csv('News_Categories.csv')

    # Initial Cleaning
    # Drop columns that are not required & adjust column names as needed
    data.drop(columns=['Unnamed: 0', 'link'], inplace=True)  

    # Handle missing values & Removing rows where text data is missing
    data.dropna(subset=['headline', 'short_description'], inplace=True)  

    # Calculate the total number of articles in the database
    total_articles = data.shape[0]

    # Count the number of articles in each category
    category_counts = data['category'].value_counts()
    unique_categories = data['category'].nunique()

    # Calculate the average number of articles per category
    average_articles_per_category = int(total_articles / unique_categories)

    # Print the counts and the average
    print(category_counts)
    print(f"Average number of articles per category: {average_articles_per_category}")

    # Visualize the counts
    plt.figure(figsize=(12, 8))
    category_counts.plot(kind='bar')
    plt.title('Number of Articles per Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=90)
    plt.show()

    # Ask user if they want to continue with further processing
    user_choice = input("Do you want to continue with further data processing? (yes/no): ")
    return user_choice.strip().lower() == 'yes', data
def clean_and_process_data(data):
    # Apply text cleaning
    data['clean_headline'] = data['headline'].apply(clean_text)
    data['clean_description'] = data['short_description'].apply(clean_text)

# Combine cleaned text columns for richer features
    data['combined_text'] = data['clean_headline'] + ' ' + data['clean_description']

    # Feature Engineering
    # Example: Length of each article
    data['text_length'] = data['combined_text'].apply(len)

    # Data Transformation
    # Remove stopwords and vectorize text
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    features = vectorizer.fit_transform(data['combined_text'])

    # Splitting the Data
    X_train, X_test, y_train, y_test = train_test_split(features, data['category'], test_size=0.3, random_state=42)

    # Print the first few rows to verify changes
    print(data.head())

def process_data(data,average_articles_per_category):
    # Merging some of the categories
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

    # Calculate new distribution after merging
    print("Categories after merging:")
    print(data['category'].value_counts())

    # Balance the dataset by downsampling or oversampling
    balanced_data = pd.DataFrame()
    for category in data['category'].unique():
        category_slice = data[data['category'] == category]
        if len(category_slice) > average_articles_per_category:
            # Downsample categories with more than the average
            downsampled_slice = category_slice.sample(n=average_articles_per_category, random_state=42)
            balanced_data = pd.concat([balanced_data, downsampled_slice], ignore_index=True)
        elif len(category_slice) < average_articles_per_category:
            # Oversample categories with less than the average
            oversampled_slice = category_slice.sample(n=average_articles_per_category, replace=True, random_state=42)
            balanced_data = pd.concat([balanced_data, oversampled_slice], ignore_index=True)
        else:
            # If the category size is equal to the average, just add it as is
            balanced_data = pd.concat([balanced_data, category_slice], ignore_index=True)

    # Shuffle the dataset
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Display the balanced distribution
    print(balanced_data['category'].value_counts())
def clean_text(text):
    text = text.lower()  # Convert to lower case
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with single space
    return text

def alternative_action():
    print("Exiting the program. Thank you!")

def main():
    user_wants_to_proceed, data = load_and_visualize_data()
    if user_wants_to_proceed:
        process_data(data, average_articles_per_category)
        clean_and_process_data(data)
    else:
        clean_and_process_data(data)

if __name__ == "__main__":
    main()
