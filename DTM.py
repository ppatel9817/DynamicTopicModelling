#!/usr/bin/env python
# coding: utf-8

'''

The code below is designed to process and analyze a corpus of text documents that are organized by year, using the Dynamic Topic Modeling (DTM) approach provided by Gensim's LdaSeqModel. This approach is particularly useful for studying how topics evolve over time in a document corpus that spans multiple years. Here's a detailed breakdown of what each part of the script does:

1. Imports and Setup:
The script imports necessary Python modules like os for interacting with the file system, Dictionary and LdaSeqModel from Gensim for topic modeling, and preprocess_string for text preprocessing.
processed_text_directory should be defined elsewhere in your code (not shown here) and points to the root directory where the text documents are stored, organized by year.

2. Collect and Preprocess Documents:
The script iterates over sorted directories named by year under processed_text_directory. For each year, it collects text files, reads them, preprocesses them using preprocess_string (which typically includes steps like tokenization, lowercasing, removing stop words, and possibly more), and groups them by year.
documents_by_year ends up as a list of lists, where each sublist contains preprocessed documents from one year.

3. Prepare Texts and Dictionary:
The script flattens the list of yearly document lists into a single list of documents (texts), which is used to create a Dictionary object. This dictionary maps each unique token in the corpus to a unique integer ID.
It then converts the list of documents (texts) into a numerical corpus using the dictionary, where each document is represented as a bag-of-words (a list of (token_id, token_count) tuples).

4. Set Time Slices for DTM:
time_slice is an array where each entry represents the number of documents in a corresponding year. This is crucial for DTM because it informs the model how the corpus is divided into distinct time periods.

5. Run Dynamic Topic Modeling (DTM):
The script initializes and runs a LdaSeqModel with the prepared corpus, dictionary, the specified number of topics (5 in this case), and the time_slice array that indicates the distribution of documents over time.
DTM allows for the analysis of how topics evolve over different time slices (years), taking into account the sequential nature of the data.

6. Output Topics for Each Time Slice:
After modeling, the script prints the topics for each year. For each year indexed by i, it retrieves and prints the dominant topics from the DTM model, providing insights into the thematic structure of the documents for that year.

Use Case:
This setup is ideal for longitudinal text data where you expect changes over time, such as analyzing annual reports, scientific articles, or news archives. DTM can reveal shifts in focus, emergence of new themes, and fading of old themes across different years, providing valuable insights for researchers or analysts studying trends in textual data.

Practical Notes:
Ensure that processed_text_directory is correctly set to the path where your documents are stored.
The preprocessing step (preprocess_string) might need customization based on the specific content and format of your documents to ensure optimal results.
Consider the computational demand of DTM, especially with large datasets and a high number of time slices or topics.
'''

import os
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldaseqmodel import LdaSeqModel
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import yake
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load SpaCy model for word vectors
nlp = spacy.load('en_core_web_md')  # Make sure the medium model is installed
keyword_extractor = yake.KeywordExtractor()

# Set the directory where the processed text files are stored
processed_text_directory = '/Users/poojanpatel/Desktop/Final_attempt/processed_text_sorted/'

# Collect documents grouped by year
documents_by_year = []
texts = []  # This will store all preprocessed documents
years = sorted(os.listdir(processed_text_directory))
for year in years:
    year_dir = os.path.join(processed_text_directory, year)
    year_documents = []
    for root, dirs, files in os.walk(year_dir):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    processed_doc = preprocess_string(file.read())
                    year_documents.append(processed_doc)
                    texts.append(processed_doc)  # Add processed documents to texts
    documents_by_year.append(year_documents)

# Initialize TF-IDF model on all documents
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_model = vectorizer.fit_transform([' '.join(doc) for doc in texts])
feature_names = vectorizer.get_feature_names_out()

# Prepare the corpus for topic modeling
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
time_slice = [len(yearly_docs) for yearly_docs in documents_by_year]

# Run DTM
ldaseq = LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=time_slice, num_topics=5)

# Prepare to collect enhanced topic labels
enhanced_labels = []

for i, year in enumerate(years):
    topics = ldaseq.print_topics(time=i)
    year_labels = []
    for topic_no, words in enumerate(topics):
        topic_words = [word for word, _ in words]
        
        # TF-IDF Labeling
        tfidf_scores = sorted(
            ((word, tfidf_model[:, vectorizer.vocabulary_[word]].sum()) for word in topic_words if word in vectorizer.vocabulary_),
            key=lambda x: x[1], 
            reverse=True
        )[:5]  # Top 5 words
        tfidf_label = ', '.join([word for word, _ in tfidf_scores])

        # Word Vectors Labeling
        docs = nlp(" ".join(topic_words))
        word_vectors = np.array([token.vector for token in docs if token.has_vector])
        avg_vector = np.mean(word_vectors, axis=0)
        similarity_scores = sorted(docs, key=lambda token: cosine_similarity([token.vector], [avg_vector])[0][0], reverse=True)
        vector_label = ', '.join([token.text for token in similarity_scores[:5]])

        # YAKE Labeling
        keywords = keyword_extractor.extract_keywords(" ".join(topic_words))
        yake_label = ', '.join([kw[0] for kw in sorted(keywords, key=lambda kw: kw[1])[:5]])

        year_labels.append([tfidf_label, vector_label, yake_label])
    
    enhanced_labels.append([year] + [label for sublist in year_labels for label in sublist])

# Create DataFrame
column_names = ['Year']
for i in range(1, 6):
    column_names += [f'Topic {i} TF-IDF', f'Topic {i} Word Vectors', f'Topic {i} YAKE']

df_labels = pd.DataFrame(enhanced_labels, columns=column_names)
df_labels.to_csv('/Users/poojanpatel/Desktop/Final_attempt/multi_labeled_dynamic_topics_by_year.csv', index=False)

print(df_labels)


'''
This is where the visualisation of the topics extracted start.

'''


import matplotlib.pyplot as plt


# Load the CSV file to check its structure
file_path = '/Users/poojanpatel/Desktop/Final_attempt/multi_labeled_dynamic_topics_by_year.csv'
data = pd.read_csv(file_path)

# Reinitialize terms_over_time and use a continuous index for filling
terms_over_time = {}
index_list = []  # List to hold the actual index positions

for i, (index, row) in enumerate(topic1_data.iterrows()):
    index_list.append(i)  # Store continuous index
    terms = row['Topic 1 TF-IDF'].split(', ')
    for term in terms:
        if term not in terms_over_time:
            terms_over_time[term] = [0] * len(topic1_data)
        terms_over_time[term][i] = 1  # Mark the term as present using the continuous index

# Plotting the term occurrences over time
fig, ax = plt.subplots(figsize=(12, 6))
years = [int(year) for year in topic1_data['Year'].tolist()]

for term, presences in terms_over_time.items():
    ax.plot(years, presences, label=term)

ax.set_title('Occurrence of Terms in Topic 1 (TF-IDF) Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Presence')
ax.legend(title='Terms', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()

# Define a function to process and plot data for each column
def process_and_plot_terms(data, column_name, base_directory):
    # Extract the 'Year' and specific column
    column_data = data[['Year', column_name]]
    
    # Ensure year data is clean and numeric
    column_data = column_data[column_data['Year'].apply(lambda x: x.isnumeric())]

    # Dictionary to store term occurrences
    terms_over_time = {}
    
    for i, (index, row) in enumerate(column_data.iterrows()):
        terms = row[column_name].split(', ')
        for term in terms:
            if term not in terms_over_time:
                terms_over_time[term] = [0] * len(column_data)
            terms_over_time[term][i] = 1  # Mark the term as present

    # Create directory for this column
    column_dir = os.path.join(base_directory, column_name.replace(' ', '_').replace('/', '_'))
    os.makedirs(column_dir, exist_ok=True)

    # Plotting and saving
    years = [int(year) for year in column_data['Year'].tolist()]
    for term, presences in terms_over_time.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(years, presences, label=term, marker='o', linestyle='-', color='b')
        ax.set_title(f'Occurrence of "{term}" in {column_name} Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Presence')
        ax.legend()
        plt.xticks(years, rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{term.replace(',', '').replace(' ', '_')}_over_time.png"
        plt.savefig(os.path.join(column_dir, plot_filename))
        plt.close()

# Directory for all plots
base_plots_directory = '/Users/poojanpatel/Desktop/Final_attempt/all_topics_plots'
os.makedirs(base_plots_directory, exist_ok=True)

# Columns to process (assuming pattern from displayed data)
topic_label_columns = [col for col in data.columns if 'Topic' in col and ('TF-IDF' in col or 'Word Vectors' in col or 'YAKE' in col)]

# Process each column
for col in topic_label_columns:
    process_and_plot_terms(data, col, base_plots_directory)

# Return the directory containing all the saved plots
base_plots_directory

import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the CSV file
file_path = '/Users/poojanpatel/Desktop/Final_attempt/multi_labeled_dynamic_topics_by_year.csv'
data = pd.read_csv(file_path)

# Define the folder to save the plots
output_folder = '/Users/poojanpatel/Desktop/Final_attempt/Topic_Plots'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Filter the dataframe to include only numeric years
data = data[data['Year'].apply(lambda x: x.isnumeric())]

# Function to plot and save graphs for each column
def plot_and_save_terms(data, column_name):
    terms_over_time = {}
    # Process terms and track their occurrence over time
    for i, terms_string in enumerate(data[column_name]):
        terms = terms_string.split(', ')
        for term in terms:
            if term not in terms_over_time:
                terms_over_time[term] = [0] * len(data)
            terms_over_time[term][i] = 1  # Mark presence of the term

    # Plotting the terms over time
    fig, ax = plt.subplots(figsize=(12, 6))
    years = [int(year) for year in data['Year'].tolist()]
    for term, presences in terms_over_time.items():
        ax.plot(years, presences, label=term)

    ax.set_title(f'Occurrence of Terms in {column_name} Over Time')
    ax.set_xlabel('Year')
    ax.set_ylabel('Presence')
    ax.legend(title='Terms', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(years, rotation=45)
    plt.tight_layout()

    # Save the plot to the specified folder
    plt.savefig(f"{output_folder}/{column_name.replace(' ', '_').replace(',', '')}.png")
    plt.close()

# Loop through each column related to topics and apply the function
for column in data.columns:
    if 'Topic' in column and 'Year' not in column:
        plot_and_save_terms(data, column)

import matplotlib.pyplot as plt
import pandas as pd
import os
import math

# Function to process and plot data for each column with grid plots
def process_and_plot_terms(data, column_name, base_directory):
    # Extract the 'Year' and specific column
    column_data = data[['Year', column_name]]
    
    # Ensure year data is clean and numeric
    column_data = column_data[column_data['Year'].apply(lambda x: x.isnumeric())]

    # Dictionary to store term occurrences
    terms_over_time = {}
    
    for i, (index, row) in enumerate(column_data.iterrows()):
        terms = row[column_name].split(', ')
        for term in terms:
            if term not in terms_over_time:
                terms_over_time[term] = [0] * len(column_data)
            terms_over_time[term][i] = 1  # Mark the term as present

    # Create directory for this column
    column_dir = os.path.join(base_directory, column_name.replace(' ', '_').replace('/', '_'))
    os.makedirs(column_dir, exist_ok=True)

    # Determine grid size
    num_terms = len(terms_over_time)
    num_cols = 3  # Define number of columns in the grid
    num_rows = math.ceil(num_terms / num_cols)

    # Plotting all terms on a grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    fig.suptitle(f'Terms in {column_name} Over Time', fontsize=16)
    years = [int(year) for year in column_data['Year'].tolist()]
    
    for ax, (term, presences) in zip(axes.flatten(), terms_over_time.items()):
        ax.plot(years, presences, label=term, marker='o', linestyle='-')
        ax.set_title(term)
        ax.set_xlabel('Year')
        ax.set_ylabel('Presence')
        ax.legend()
        ax.set_xticks(years)
        ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout and save the grid plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    grid_filename = f"{column_name.replace(' ', '_').replace('/', '_')}_grid_plot.png"
    plt.savefig(os.path.join(column_dir, grid_filename))
    plt.close()

# Directory for all plots
base_plots_directory = '/Users/poojanpatel/Desktop/Final_attempt/all_grid_topics_plots'
os.makedirs(base_plots_directory, exist_ok=True)

# Columns to process
topic_label_columns = [col for col in data.columns if 'Topic' in col and 'Year' not in col]

# Process each column
for col in topic_label_columns:
    process_and_plot_terms(data, col, base_plots_directory)

