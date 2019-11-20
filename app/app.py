
import json
import pandas as pd
import plotly
import re
from collections import Counter

# import NLP libraries
from tokenizer_function import Tokenizer, tokenize

from flask import Flask
from plotly.graph_objs import Bar
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


@app.before_first_request

def model_data_loader():
    global table_df
    global model_pkl

    db_engine = create_engine('sqlite:///data/DisasterResponse.db')
    table_df = pd.read_sql_table('DisasterResponse', db_engine)

    model_pkl = joblib.load("models/adaboost_model.pkl")

@app.route('/')
@app.route('/index')

def index():

    genre_item_counts = table_df.groupby('genre').count()['message']
    genre_item_names = list(genre_item_counts.index)

    category_counts_df = table_df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_counts = list(category_counts_df)
    category_names = list(category_counts_df.index)

    sm_messages = ' '.join(table_df[table_df['genre'] == 'social']['message'])
    sm_tokens = tokenize(sm_messages)
    sm_word_counter = Counter(sm_tokens).most_common()
    sm_word_count = [i[1] for i in sm_word_counter]
    sm_word_pct = [i/sum(sm_word_count) *100 for i in sm_word_count]
    sm_words = [i[0] for i in sm_word_counter]

    direct_message = ' '.join(table_df[table_df['genre'] == 'direct']['message'])
    direct_token = tokenize(direct_message)
    direct_word_counter = Counter(direct_token).most_common()
    direct_word_count = [i[1] for i in direct_word_counter]
    direct_word_pct = [i/sum(direct_word_count) * 100 for i in direct_word_count]
    direct_words = [i[0] for i in direct_word_counter]

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_item_names,
                    y=genre_item_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # histogram of social media messages top 30 keywords 
        {
            'data': [
                    Bar(
                        x=sm_words[:50],
                        y=sm_word_pct[:50]
                                    )
            ],

            'layout':{
                'title': "Top 50 Keywords in Social Media Messages",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "% Total Social Media Messages"    
                }
            }
        }, 

        # histogram of direct messages top 30 keywords 
        {
            'data': [
                    Bar(
                        x=direct_words[:50],
                        y=direct_word_pct[:50]
                                    )
            ],

            'layout':{
                'title': "Top 50 Keywords in Direct Messages",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "% Total Direct Messages"    
                }
            }
        }, 



        {
            'data': [
                    Bar(
                        x=category_names,
                        y=category_counts
                                    )
            ],

            'layout':{
                'title': "Distribution of Message Categories",
                'xaxis': {'tickangle':60
                },
                'yaxis': {
                    'title': "count"    
                }
            }
        },     

    ]
    
    ip_ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    ip_graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ip_ids, graphJSON=ip_graph_json)

@app.route('/go')
def go():

    user_query = request.args.get('query', '')

    classify_labels = model_pkl.predict([user_query])[0]
    classify_results = dict(zip(table_df.columns[4:], classify_labels))

    return render_template(
        'go.html',
        query=user_query,
        classification_result=classify_results
    )


def main():
    app.run()


if __name__ == '__main__':
    main()
