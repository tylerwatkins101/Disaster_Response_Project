import json
import plotly
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    ''' tokenize raw text data before tf-idf transformation '''
    tokens = RegexpTokenizer(r'\w+').tokenize(text)
    less_tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in less_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('labelled_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for original visual
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract data needed for visual 1
    df_critical = df[['food','water','shelter','medical_help','missing_people','death']]
    critical_percentages = round(df_critical.apply(np.mean).sort_values(ascending=False)*100,2)
    critical_names = list(critical_percentages.index)

    # extract data needed for visual 2
    df_breakdowns = df[['related','request','offer']]
    breakdown_totals = df_breakdowns.apply(sum).sort_values(ascending=False)
    breakdown_names = list(breakdown_totals.index)
    breakdown_totals = list(breakdown_totals.values)
    breakdown_totals.insert(0,len(df))
    breakdown_names.insert(0,'total')
    breakdown_names[1] = 'related_to_event'

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=breakdown_names,
                    y=breakdown_totals
                )
            ],

            'layout': {
                'title': 'Messages Received During Event',
                'yaxis': {
                    'title': "Total"
                },
                'xaxis': {
                    'title': "Major Message Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=critical_names,
                    y=critical_percentages
                )
            ],

            'layout': {
                'title': 'Percentage of Messages Related to Critical Needs',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Critical Message Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
