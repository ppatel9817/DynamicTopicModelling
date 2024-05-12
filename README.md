# Dynamic Topic Modeling (DTM) for Longitudinal Text Data

Overview
This repository contains a Python script to process and analyze a corpus of text documents organized by year. The script employs Dynamic Topic Modeling (DTM) provided by Gensim's LdaSeqModel, which is effective for studying how topics evolve over time. This approach is especially beneficial for examining documents spanning multiple years, such as scholarly articles, corporate reports, or legislative documents.

Features
Dynamic Topic Analysis: Gensim's LdaSeqModel is used to understand how topics change yearly.
Automated Preprocessing: Includes text preprocessing using preprocess_string from Gensim.
Enhanced Labeling: Labels topics using TF-IDF, word vectors, and YAKE keyword extraction for enhanced interpretability.
Visualization: Provides basic plotting of topic evolution over time.

The script will:

Read and preprocess text data from files organized by year.
Perform DTM to analyze the evolution of topics across specified time slices.
Output the results as labeled topics and visualize the topic trends.
Structure of the Script
Imports and Setup: Libraries needed for file handling, topic modeling, text preprocessing, and plotting are imported.

Document Collection and Preprocessing: Text files are read and preprocessed year-wise from a specified directory.

Texts and Dictionary Preparation: A corpus for topic modeling is prepared, and each document is converted to a bag-of-words model.

Dynamic Topic Modeling: The LdaSeqModel is applied to the corpus with time slices indicating the distribution of documents per year.

Output Topics and Visualization: Topics for each time slice are printed, and key terms are visualized over time to show their evolution.

Output
The script prints topics detected for each year directly to the console.
A CSV file with multi-labeled dynamic topics is saved.
Basic plots showing the trend of terms within topics over years are generated.
Example Visualization
To visualize the trends of specific terms within topics over time, the script plots these trends using matplotlib. These visualizations help in understanding the prominence and decline of themes within the corpus over the examined period.

Customization
Modify time slices, number of topics, or preprocessing steps according to your dataset and requirements.
Extend visualization functions to create more sophisticated plots or integrate with web-based visualization tools like Plotly or Dash for interactive exploration.
