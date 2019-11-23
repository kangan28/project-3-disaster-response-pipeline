import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from models.tokenizer_function import Tokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.externals import joblib

def data_to_load(database_filepath):
    engine_in_database = create_engine('sqlite:///DisasterResponse.db')
    dataframe_in_table = pd.read_sql_table('DisasterResponse', engine_in_database)
    X = dataframe_in_table['message']
    Y = dataframe_in_table.drop(['id', 'original', 'message', 'genre'], axis=1)
    labels_of_category = Y.columns
    return labels_of_category, X, Y

def model_to_build():
    pipe_to_be_made = Pipeline([
        ('tokenizer', Tokenizer()),
        ('vec', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100)))
    ])
    params_in_search = {'clf__estimator__max_features':['sqrt', 0.5],
              'clf__estimator__n_estimators':[50, 100]}
    cross_validator = GridSearchCV(estimator=pipe_to_be_made, param_grid = params_in_search, cv = 5, n_jobs = 10)
    return cross_validator

def model_to_evaluate(model, X_test, Y_test, category_with_name):
    y_predval = model.predict(X_test)

    detail_with_perform = []
    for i in range(len(category_with_name)):
        detail_with_perform.append([f1_score(Y_test.iloc[:, i].values, y_predval[:, i], average='micro'),
                                    precision_score(Y_test.iloc[:, i].values, y_predval[:, i], average='micro'),
                                    recall_score(Y_test.iloc[:, i].values, y_predval[:, i], average='micro')])

    detail_with_perform = pd.DataFrame(detail_with_perform, columns=['f1 score', 'precision', 'recall'],
                                       index = category_with_name)
    return detail_with_perform

def model_to_save(model, model_fp):

    joblib.dump(model, open(model_fp, 'wb'))

def main():
    if len(sys.argv) == 3:
        db_fp, model_fp = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(db_fp))
        X, Y, category_names = data_to_load(db_fp)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        print('Building model...')
        model = model_to_build()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        model_to_evaluate(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_fp))
        model_to_save(model, model_fp)

        print('Trained model saved!')

    else:
        print('provide path of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()