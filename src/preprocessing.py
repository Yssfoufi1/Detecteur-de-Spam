import re
import pandas as pd


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df = df.drop_duplicates(subset='text').reset_index(drop=True)
    df['clean_text'] = df['text'].apply(clean_text)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    return df