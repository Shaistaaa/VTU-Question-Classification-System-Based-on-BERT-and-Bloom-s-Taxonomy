# -*- coding: utf-8 -*-


import gensim.downloader as api
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load pre-trained Word2Vec model
word_vect = api.load('word2vec-google-news-300')

# Load spaCy tokenizer
nlp = spacy.load('en_core_web_md')

# Tokenizes a sentence using spaCy library
def spacy_tokenizer(text):
    doc = nlp(text)

    # Lemmatize tokens
    tokens = [word.lemma_.strip() for word in doc]

    return tokens

# Takes a tokenized sentence input from spaCy and computes the average of all word vectors
def sentence_vect(tokens):
    # Word2Vec output size (300 for word2vec-google-news-300)
    size = word_vect.vector_size

    # Create a vector of zeros
    sent_vect = np.zeros(size)

    # Get the average of all word vectors of each token
    counter = 0
    for word in tokens:
        if word in word_vect:
            sent_vect += word_vect[word]
            counter += 1

    if counter == 0:
        return None

    sent_vect = sent_vect / counter
    return sent_vect

# Read the dataset
df = pd.read_csv('blooms_dataset.csv')

# Remove all punctuations and change to lowercase
df['Text'] = df['Text'].str.replace(r'[^\w\s]+', '', regex=True)
df['Text'] = df['Text'].str.lower()

# Encode categories into numerical values
df['Label'] = pd.factorize(df.Label)[0]

# Blooms taxonomy categories
categories = ['Analyse', 'Apply', 'Create', 'Evaluate', 'Remember', 'Understand']

# Add a column of tokenized text
df['Tokens'] = df['Text'].apply(spacy_tokenizer)

# Create sentence vectors by taking the average of token vectors
df['Vectors'] = df['Tokens'].apply(sentence_vect)

# Drop all null values after tokenizing as some cells may not have relevant data
df = df.dropna(axis=0, subset=['Vectors'])

# Split data into training and test sets (80% training, 20% test)
train, test = train_test_split(df, test_size=0.2)

"""## Model Building"""

# Create a Support Vector Machine (SVM) model
model = SVC(kernel='linear', gamma='auto', degree=3)

# Fit sentence vectors to labels
model.fit(list(train.Vectors), train.Label)

# Generate Blooms labels on the test set
labels = model.predict(list(test.Vectors))

# Model report
# Heatmap of the model
mat = confusion_matrix(test.Label, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=categories, yticklabels=categories)

plt.xlabel('True label')
plt.ylabel('Predicted label')

print(classification_report(test.Label, labels))

# Model Accuracy: How often is the classifier correct?
acc = accuracy_score(test.Label, labels)
print("Accuracy:", acc)

"""## Prediction Testing"""

category_dict = {0: 'Analyse', 1: 'Apply', 2: 'Create', 3: 'Evaluate', 4: 'Remember', 5: 'Understand'}

def predict_blooms(text, model):
    process = spacy_tokenizer(text)
    process = sentence_vect(process)
    blooms = model.predict([process])
    return category_dict[blooms[0]]

print("Available categories:", categories)

while True:
    task = input("\nEnter a task or 'exit' to quit: ")
    
    if task.lower() == 'exit':
        break

    predicted_category = predict_blooms(task, model)
    print("Predicted Class:", predicted_category)
