import sys
import pandas as pd
import numpy as n
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    # load messages and categories
    input_messages = pd.read_csv(messages_filepath)
    message_categories = pd.read_csv(categories_filepath)
    
    merged_df = pd.merge(input_messages, message_categories, on = 'id')
    return merged_df

def clean_data(df):
    
    split_categories = df['categories'].str.split(';', expand=True)
    
    row = split_categories.iloc[1]
    cols_category = row.apply(lambda x: x.split('-')[0])
    split_categories.columns = cols_category
    
    for item in split_categories:
        split_categories[item] = split_categories[item].apply(lambda x: int(x.split('-')[1]))
    df.drop('categories', axis = 1, inplace = True)
    df = df.join(split_categories)
    
    df = df.drop_duplicates()
    return df
    

def save_data(df, database_filename):
    
    db_engine = create_engine('sqlite:///Messages.db')
    df.to_sql('Messages', db_engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_fp, categories_fp, db_fp = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_fp, categories_fp))
        df = load_data(messages_fp, categories_fp)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(db_fp))
        save_data(df, db_fp)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
