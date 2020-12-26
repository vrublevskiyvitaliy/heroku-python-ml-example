# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, url_for

import json
import logging
import os

from sklearn.externals import joblib
from werkzeug.utils import secure_filename

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(APP_ROOT, 'classifier.pkl')

PORT = 5000

app = Flask(__name__)
logging.basicConfig(filename='movie_classifier.log', level=logging.DEBUG)
model = joblib.load(MODEL)
label = {0: 'negative', 1: 'positive'}


# @app.route('/')
# def home():
#     return 'It works.'


def predict(model, text):
    return label[model.predict([text])[0]]


@app.route('/review', methods=['POST'])
def extract():
    """Return the movie review sentiment score.
    
    Returns a JSON object :
    {
         "sentiment": "positive"
    }
    """
    if request.method == 'POST':
        description = request.form.get('text', '')

        result = {
            'sentiment': predict(model, description)
        }
        # return json.dumps(result)
        return render_template('result.html', prediction=result['sentiment'])


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
    similarity = 89
    return render_template('compare-sentences.html', first_sentence=first_sentence,
                           second_sentence=second_sentence, similarity=similarity)


@app.route('/compare-files')
def compare_files():
    first_sentence = request.form.get('first-sentence', '')
    second_sentence = request.args.get('second-sentence', '')
    similarity = 89
    return render_template('compare-sentences.html', first_sentence=first_sentence,
                           second_sentence=second_sentence, similarity=similarity)




# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         message = request.form['message']
#         data = [message]
#         vect = pd.DataFrame(cv.transform(data).toarray())
#         body_len = pd.DataFrame([len(data) - data.count(" ")])
#         punct = pd.DataFrame([count_punct(data)])
#         total_data = pd.concat([body_len, punct, vect], axis=1)
#         my_prediction = clf.predict(total_data)
#     return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
