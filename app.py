import os
import requests
import pandas as pd
from flask import Flask, request, render_template, session, flash
from bs4 import BeautifulSoup
from joblib import load

app = Flask(__name__)

tfidf_vectorizer = load('artifacts/tfidf_vectorizer.joblib')
tfidf_model = load('artifacts/tfidf_train.joblib')
pac = load('artifacts/pac.joblib')


@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('index.html')


@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'password' and request.form['username'] == 'admin':
        session['logged_in'] = True
    return home()


@app.route('/predict', methods=['POST'])
def predict():
    res = requests.get(request.form.get('url'))
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text=True)

    output = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
    ]

    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)

    output = pd.Series(output)
    vectorized_output = tfidf_vectorizer.transform(output)
    prediction = pac.predict(vectorized_output)

    return render_template('index.html', prediction_text='This is {} news!'.format(prediction[0]))


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True) # Make sure to change this to false before deployment
