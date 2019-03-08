import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sqlalchemy import create_engine
import pickle

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def load_data(database_filepath):
    ''' load data from database '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM labelled_messages', engine)
    X = df.message
    Y = df.drop(['id','message','original','genre'], axis = 1)
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    ''' tokenize text data for tf-idf transformation '''
    tokens = RegexpTokenizer(r'\w+').tokenize(text)
    less_tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in less_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Pipeline for building classification model
    
    Parameters for model selected using GridSearchCV as follows:
    parameters = {
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__min_samples_split': [3, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring = 'f1_weighted')
    cv.fit(X,Y)
    cv.best_params_

    Code not run here because it takes over an hour to run on a single machine
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, min_samples_split=4)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Evaluate model and print precision, recall and f1-score for each category '''
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category: "+category_names[i])
        print(classification_report(Y_test.values[:,i], pd.DataFrame(Y_pred)[i]))


def save_model(model, model_filepath):
    ''' Save model to pickle file '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
