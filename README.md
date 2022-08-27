# Automated Extraction of Quotations From Text

A quotation is a piece of saying or writing that strikes us so true or memorable that it is quoted by others and used in speech or writing. It is authoritative, motivating, inspiring, entertaining, or attention-grabbing.

However, these text qualities are not limited to just the sayings of famous people. Anyone can generate content that potentially can be used as quotations too. What if there is a system to extract potential quotable quotes from any text given?

We first build a dataset of text consisting of sentences, train machine learning and deep learning models to predict which are quotations and which are not. Then we deploy the most accurate model an Application Programming Interface (API) to be used by any applications to extract quotations from text.

Contact me if you like to see how it works!

<a href="https://github.com/rickysoo/nlp-quote/blob/main/quote_prepare.ipynb">quote_prepare.ipnyb</a> - Ingest data and build a dataset of texts.  
<a href="https://github.com/rickysoo/nlp-quote/blob/main/quote_train.ipynb">quote_train.ipynb</a> - Train models to predict quotability of texts.  
<a href="https://github.com/rickysoo/nlp-quote/blob/main/quote_deploy.ipynb">quote_deploy.ipynb</a> - Deploy model to extract quotations from any text.  

Keywords: Natural Language Processing, Quotations, Text Classification, Text Extraction, Prediction Model.

## Updates - August 2022

A new RNN model is trained using the best hyperparameters found using Random Search. Then the new model is deployed to be run online and on local computer.

<a href="https://github.com/rickysoo/nlp-quote/blob/main/quote_rnn.ipynb">quote_rnn.ipynb</a> - Search for the best hyperparameters for a better RNN model.  
<a href="https://github.com/rickysoo/nlp-quote/blob/main/quote_deploy2.ipynb">quote_deploy2.ipynb</a> - Deploy model to extract quotations using the new model.  
<a href="https://github.com/rickysoo/nlp-quote/blob/main/server.py">server.py</a> - Run the API server on local computer.  
