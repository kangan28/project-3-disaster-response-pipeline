import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(tokenize).values


def tokenize(text):
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    matched_url_patterns = re.findall(url_regex, text)
    for url_items in matched_url_patterns:
        text = text.replace(url_items, "urlplaceholder")
    tokenized_items = word_tokenize(text)
    lemmatizer_instance = WordNetLemmatizer()

    cleaned_tokens = [lemmatizer_instance.lemmatize(tok).lower().strip() for tok in tokenized_items]

    STOPWORDS = list(set(stopwords.words('english')))
    cleaned_tokens = [token for token in cleaned_tokens if token not in STOPWORDS]
    
    return cleaned_tokens
