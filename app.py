# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, url_for
from model import features_for_prediction

import logging
import os

from sklearn.externals import joblib

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_V = os.path.join(APP_ROOT, 'finalized_model.sav')

PORT = 5000

app = Flask(__name__)
logging.basicConfig(filename='classifier.log', level=logging.DEBUG)

model_v = joblib.load(MODEL_V)


def predict_v(s1, s2):
    features = features_for_prediction(s1, s2)
    is_paraphrase = model_v.predict(features)[0] == 1
    probabilities = model_v._predict_proba_lr(features)

    return {
        'is_paraphrase': is_paraphrase,
        'not_paraphrase_probability': int(round(probabilities[0][0] * 100)),
        'paraphrase_probability': int(round(probabilities[0][1] * 100)),
    }
    # return {
    #     'is_paraphrase': 0,
    #     'not_paraphrase_probability': 0,
    #     'paraphrase_probability': 0,
    # }


@app.route('/')
def test2():
    return render_template('m_index.html')


@app.route('/compare-sentences')
def compare_sentences():
    first_sentence = request.args.get('first-sentence', '')
    second_sentence = request.args.get('second-sentence', '')
    similarity = predict_v(first_sentence, second_sentence)
    return render_template('compare-sentences.html', first_sentence=first_sentence,
                           second_sentence=second_sentence, similarity=similarity)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
