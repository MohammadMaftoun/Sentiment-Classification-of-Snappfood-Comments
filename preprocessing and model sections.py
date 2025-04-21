# Install required libraries for text processing and machine learning
!pip install hazm

# Install additional libraries for advanced machine learning models and visualization
!pip install catboost wordcloud xgboost lightgbm

# Import necessary libraries for data manipulation, text processing, and modeling
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Import text processing tools from the hazm library for Persian language
from hazm import Normalizer, Stemmer, word_tokenize, Lemmatizer, stopwords_list
# Import libraries for word embeddings
from gensim.models import Word2Vec, FastText
# Import evaluation and data splitting utilities
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# Import machine learning models
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
# Import pipeline and preprocessing tools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Import visualization tool for word clouds
from wordcloud import WordCloud

# Download dataset from Google Drive
!gdown --id '1HZ8HDQXnI_8A0wP9SnSwxTalEtv6ECM-'

# Load the Snappfood dataset into a pandas DataFrame
df = pd.read_csv('/content/Snappfood - Sentiment Analysis.csv')
# Display the first few rows of the dataset
df.head()

# Check the dimensions of the dataset
df.shape

# Display summary information about the dataset
df.info()

# Provide descriptive statistics for all columns
df.describe(include='all')

# Count the occurrences of each label
df['label'].value_counts()

# Show the number of unique values in each column
df.nunique()

# Select relevant columns (comment, label, label_id) and remove rows with missing values
df = df[['comment', 'label', 'label_id']]
df.dropna(inplace=True)
# Display the first few rows after cleaning
df.head()

# Verify if there are any rows where the label 'HAPPY' is incorrectly assigned label_id = 1.0
# Note: 'HAPPY' should have label_id = 0.0, and 'SAD' should have label_id = 1.0
df.query('label == "HAPPY" and label_id == 1.0')

# Check for rows where the label 'SAD' is incorrectly assigned label_id = 0.0
df.query('label == "SAD" and label_id == 0')

# Convert label_id column to integer type
df['label_id'] = df['label_id'].astype(int)
# Display the first few rows to confirm changes
df.head()

# Separate features (comments) and target (label_id)
X = df['comment']
y = df['label_id']

# Verify the shape of the feature set
X.shape

# Clean the text by removing punctuation, English letters, and digits to keep only Persian text
X = X.apply(lambda x: re.sub(r'[\da-zA-Z\!\(\)\-\[\]\{\}\;\:\'\"\\\,\<\>\.\/\?\@\#\$\%\^\&\*\_\~\؟\،\٪\×\÷\»\«]', '', x))

# Display the first few rows of cleaned text
X.head()

# Split the data into training and testing sets with a 25% test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Normalize the text using hazm's Normalizer to standardize Persian characters
normalizer = Normalizer()
X_train = X_train.apply(lambda x: normalizer.normalize(x))
X_test = X_test.apply(lambda x: normalizer.normalize(x))

# Display the first 10 rows of normalized training data
X_train.head(10)

# Tokenize the text into individual words using hazm's word_tokenize
X_train = X_train.apply(lambda x: word_tokenize(x))
X_test = X_test.apply(lambda x: word_tokenize(x))

# Display the first few rows of tokenized test data
X_test.head()

# Apply stemming to reduce words to their root form using hazm's Stemmer
stemmer = Stemmer()
X_train = X_train.apply(lambda words: [stemmer.stem(word) for word in words])
X_test = X_test.apply(lambda words: [stemmer.stem(word) for word in words])

# Display the first few rows of stemmed test data
X_test.head()

# Apply lemmatization to further normalize words to their base form
lemmatizer = Lemmatizer()
X_train = X_train.apply(lambda words: [lemmatizer.lemmatize(word) for word in words])
X_test = X_test.apply(lambda words: [lemmatizer.lemmatize(word) for word in words])

# Display the first few rows of lemmatized training data
X_train.head()

# Join the list of words in each row back into a single string
X_train = X_train.apply(lambda x: ' '.join(x)).to_list()
X_test = X_test.apply(lambda x: ' '.join(x)).to_list()

# Convert text data into TF-IDF features for model training
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(X_train)
x_test = vectorizer.transform(X_test)

# Train and evaluate a Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
# Print performance metrics for the Random Forest model
print(classification_report(y_test, y_pred))

# Train and evaluate a CatBoost Classifier
cat = CatBoostClassifier()
cat.fit(x_train, y_train)
y_pred = cat.predict(x_test)
# Print performance metrics for the CatBoost model
print(classification_report(y_test, y_pred))

# Train and evaluate a LightGBM Classifier
lgbm = lgb.LGBMClassifier()
lgbm.fit(x_train, y_train)
y_pred = lgbm.predict(x_test)
# Print performance metrics for the LightGBM model
print(classification_report(y_test, y_pred))

# Train and evaluate an XGBoost Classifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
# Print performance metrics for the XGBoost model
print(classification_report(y_test, y_pred))

# Import additional libraries for ensemble modeling
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# Define pipelines for each classifier to include TF-IDF vectorization
xgb_pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('xgb', XGBClassifier(random_state=42, eval_metric='logloss'))
])

catboost_pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('catboost', CatBoostClassifier(random_state=42, verbose=0))
])

lgbm_pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('lgbm', lgb.LGBMClassifier(random_state=42))
])

# Create a Voting Classifier to combine predictions from multiple models
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_pipeline),
        ('catboost', catboost_pipeline),
        ('lgbm', lgbm_pipeline)
    ],
    voting='soft'  # Use soft voting to average probabilities
)

# Train the Voting Classifier on the training data
voting_clf.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = voting_clf.predict(X_test)

# Print performance metrics for the Voting Classifier
print(classification_report(y_test, y_pred))

# Test the model with a sample Persian comment
voting_clf.predict(['خیلی خوشمزه بود'])

# Save the trained Voting Classifier model to a file
import pickle
with open('voting_clf.pkl', 'wb') as file:
    pickle.dump(voting_clf, file)

# Save the TF-IDF vectorizer to a file for future use
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)