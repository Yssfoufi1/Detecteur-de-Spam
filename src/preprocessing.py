# Importation des modules necessaires
import re
import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Telechargement des ressources NLTK (tokenization, POS tagging, lemmatisation)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
lemmatizer = WordNetLemmatizer()

# Convertit une etiquette grammaticale POS vers le format attendu par WordNet
def get_wordnet_pos(tag):
    if tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Nettoie et normalise un texte : minuscule, ponctuation, tokenization, lemmatisation
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    return ' '.join(lemmatized)

# Liste d'expressions typiques du phishing / credential harvesting
PHISHING_KEYWORDS = [
    'verify your account', 'confirm your account', 'verify your identity',
    'reset your password', 'password will expire', 'update your password',
    'account will be suspended', 'account has been suspended', 'account is locked',
    'unusual activity', 'suspicious activity', 'unauthorized access',
    'click here', 'login now', 'sign in now',
    'urgent action', 'immediate action', 'act now', 'action required',
    'update your information', 'update your payment', 'confirm your payment',
    'security alert', 'security notice', 'security warning', ]

# Compte le nombre d'expressions de phishing presentes dans un texte
def phishing_keyword_score(text):
    text_lower = text.lower()
    return sum(1 for phrase in PHISHING_KEYWORDS if phrase in text_lower)

# Extrait des features numeriques (longueur, majuscules, ponctuation...) en plus du TF-IDF
def extract_extra_features(text_series):
    df_feat = pd.DataFrame()
    df_feat['length'] = text_series.apply(len)
    df_feat['uppercase_ratio'] = text_series.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1) )
    df_feat['exclamation_count'] = text_series.apply(lambda t: t.count('!'))
    df_feat['digit_count'] = text_series.apply(lambda t: sum(c.isdigit() for c in t))
    df_feat['has_url'] = text_series.apply(
        lambda t: 1 if re.search(r'http|www\.|\.com', t.lower()) else 0 )
    df_feat['phishing_keyword_score'] = text_series.apply(phishing_keyword_score)
    return df_feat

# Charge un CSV (format brut Kaggle ou deja fusionne), nettoie et prepare les donnees
def load_and_prepare_data(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'text']
    else:
        df = df[['label', 'text']]
    df = df.drop_duplicates(subset='text').reset_index(drop=True)
    df['clean_text'] = df['text'].apply(clean_text)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    return df