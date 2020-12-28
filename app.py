# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, url_for
from model import features_for_prediction

import json
import logging
import os
from sklearn.svm import LinearSVC

from sklearn.externals import joblib
from werkzeug.utils import secure_filename

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_V = os.path.join(APP_ROOT, 'finalized_model.sav')

PORT = 5000

app = Flask(__name__)
logging.basicConfig(filename='movie_classifier.log', level=logging.DEBUG)

model_v = joblib.load(MODEL_V)

label = {0: 'negative', 1: 'positive'}


# @app.route('/')
# def home():
#     return 'It works.'



def predict_v(model_v, s1, s2):
    features = features_for_prediction(s1, s2)
    is_paraphrase = model_v.predict(features)[0] == 1
    probabilities = model_v._predict_proba_lr(features)
    return {
        'is_paraphrase': is_paraphrase,
        'not_paraphrase_probability': probabilities[0][0],
        'paraphrase_probability': probabilities[0][1],
    }


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/test')
def test():
    return render_template('material_index.html')


@app.route('/test2')
def test2():
    return render_template('m_index.html')


@app.route('/compare-sentences')
def compare_sentences():
    first_sentence = request.args.get('first-sentence', '')
    second_sentence = request.args.get('second-sentence', '')
    similarity = predict_v(model_v, first_sentence, second_sentence)['paraphrase_probability']
    return render_template('compare-sentences.html', first_sentence=first_sentence,
                           second_sentence=second_sentence, similarity=similarity)


@app.route('/compare-files')
def compare_files():
    first_sentence = request.form.get('first-sentence', '')
    second_sentence = request.args.get('second-sentence', '')
    similarity = 89
    return render_template('compare-sentences.html', first_sentence=first_sentence,
                           second_sentence=second_sentence, similarity=similarity)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
