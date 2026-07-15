import re #importation du module re pour les expressions régulières
import nltk #importation du module nltk pour le traitement du langage naturel
import pandas as pd #importation du module pandas pour la manipulation de données
from nltk import pos_tag #importation de la fonction pos_tag pour l'étiquetage grammatical
from nltk.corpus import wordnet #importation du module wordnet pour la lemmatisation
from nltk.stem import WordNetLemmatizer #importation de la classe WordNetLemmatizer pour la lemmatisation
from nltk.tokenize import word_tokenize  #importation de la fonction word_tokenize pour la tokenisation

#telechargement des ressources nécessaires pour lemmatisation et tokenization
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('own-1.4', quiet=True)
lemmatizer = WordNetLemmatizer() #création d'une instance de la classe WordNetLemmatizer

# Fonction pour obtenir la catégorie grammaticale de WordNet à partir de l'étiquette POS
def get_wordnet_pos(tag):
    if tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text) #tokenisation du texte en mots
    tagged = pos_tag(tokens) #identification de la catégorie grammaticale de chaque mot
    lemmatized = [lemmatizer.lemmatize(w , get_wordnet_pos(t)) for w, t in tagged] #lemmatisation des mots en utilisant la catégorie grammaticale appropriée
    return ' '.join(lemmatized)



def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df = df.drop_duplicates(subset='text').reset_index(drop=True)
    df['clean_text'] = df['text'].apply(clean_text)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    return df
