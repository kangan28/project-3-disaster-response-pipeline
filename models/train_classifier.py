import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from models.tokenizer_function import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

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
    
def data_to_load(database_filepath):
    engine_in_database = create_engine('sqlite:///DisasterResponse.db')
    dataframe_in_table = pd.read_sql_table('DisasterResponse', engine_in_database)
    X = dataframe_in_table['message']
    Y = dataframe_in_table.drop(['id', 'original', 'message', 'genre'], axis=1)
    labels_of_category = Y.columns
    return labels_of_category, X, Y

