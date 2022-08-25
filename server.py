import pandas as pd
import numpy as np
import tensorflow

from tensorflow.keras import preprocessing as kprocessing
from tensorflow.keras.models import load_model

import nltk
from nltk.tokenize import sent_tokenize

import pickle

from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, jsonify

sentences_test = [
  'A journey of a thousand miles begins with a single step.',
  'Are you beginning a journey of a thousand steps tomorrow?',
  'The Post cited people familiar with the investigation as saying that federal agents were looking for classified documents related to nuclear weapons',
  'But it turns out that Trump did take such material from the White House.',
  'No one will reap except what they sow.'
]


def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision


def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall


model = load_model('rnn2.h5', custom_objects={'precision_m': precision_m, 'recall_m': recall_m})
corpus = pickle.load(open('corpus.model', 'rb'))

max_words = 30000
max_len = 200

tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', num_words=max_words, oov_token="<pad>", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(corpus)


def sent_predict(sentences, tokenizer=tokenizer, threshold=0.5):
  seq = kprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(sentences), maxlen=max_len)
  predictions = model.predict(seq).reshape(-1)
  predictions = dict(zip(sentences, predictions))
  predictions = { key:float(value) for (key, value) in predictions.items() if value >= threshold }
  return predictions


def extract_quotes(text, threshold=0.5):
  return sent_predict(sent_tokenize(text), threshold=threshold)

output_test = sent_predict(sentences_test)


app = Flask(__name__)
run_with_ngrok(app)

@app.route('/', methods=['GET', 'POST'])
def home():
  return jsonify(output_test)

@app.route('/api', methods=['POST'])
def api():
  if request.form.get('data'):
    data = request.form['data']
    threshold = 0.5

    if request.form.get('threshold'):
      th = request.form['threshold']

      try:
        th = float(th)
        
        if th >= 0.0 and th <= 1.0:
          threshold = th

      except ValueError:
        pass

    quotes = extract_quotes(data, threshold=threshold)
    return quotes
  else:
    return 'Please include text in the POST request.', 401

if __name__ == '__main__':
  app.run()
