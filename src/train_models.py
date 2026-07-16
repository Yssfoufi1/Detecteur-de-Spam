from pathlib import Path #importation de la bibliothèque pathlib pour gérer les chemins de fichiers
import joblib #importation de la bibliothèque joblib pour la sérialisation des modèles
from sklearn.model_selection import GridSearchCV, train_test_split #importation de la fonction train_test_split pour diviser les données en ensembles d'entraînement et de test
from sklearn.feature_extraction.text import TfidfVectorizer #importation de la classe TfidfVectorizer pour la vectorisation des textes
from sklearn.naive_bayes import MultinomialNB #importation de la classe MultinomialNB pour le modèle de classification Naive Bayes multinomial
from sklearn.linear_model import LogisticRegression #importation de la classe LogisticRegression pour le modèle de régression logistique
from sklearn.svm import SVC #importation de la classe SVC pour le modèle de machine à vecteurs de support
from sklearn.ensemble import RandomForestClassifier #importation de la classe RandomForestClassifier pour le modèle de forêt aléatoire
from preprocessing import load_and_prepare_data #importation de la fonction load_and_prepare_data pour charger et préparer les données

BASE_DIR = Path(__file__).resolve().parent.parent #définition du répertoire de base du projet
DATA_PATH = BASE_DIR / 'data' / 'spam.csv' #définition du chemin vers le fichier de données
MODELS_DIR = BASE_DIR / 'models' #définition du répertoire pour enregistrer les modèles
MODELS_DIR.mkdir(exist_ok=True) #création du répertoire pour les modèles s'il n'existe pas

df = load_and_prepare_data(DATA_PATH) #chargement et préparation des données
X = df['clean_text'] #sélection des textes nettoyés comme caractéristiques
y = df['target'] #sélection des étiquettes comme cibles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #division des données en ensembles d'entraînement et de test

vectorizer = TfidfVectorizer(stop_words='english') #création d'une instance de TfidfVectorizer
X_train_tfidf = vectorizer.fit_transform(X_train) #vectorisation des textes d'entraînement

param_grids = {
    'naive_bayes': {'alpha': [0.1, 0.5, 1.0, 2.0]},
    'logistic_regression': {'C': [0.1, 1, 10]},
    'svm': {'C': [0.1, 1, 10]},
    'random_forest': {'n_estimators': [100, 200], 'max_depth': [None, 30]},
}

models = {
    'naive_bayes' : MultinomialNB(), #modèle Naive Bayes multinomial
    'logistic_regression' : LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42), #modèle de régression logistique
    'svm' : SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42), #modèle de machine à vecteurs de support avec noyau linéaire
    'random_forest' : RandomForestClassifier(class_weight='balanced', random_state=42), #modèle de forêt aléatoire avec pondération des classes
}

for name, model in models.items():
    print(f"\nGridSearch : {name} ...")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train_tfidf, y_train)
    print(f"  Meilleurs parametres : {grid.best_params_}")
    print(f"  Meilleur F1 (validation croisee) : {grid.best_score_:.4f}")
    best_model = grid.best_estimator_

    if name == 'svm':
        best_C = grid.best_params_['C']
        best_model = SVC(kernel='linear', C=best_C, class_weight='balanced', probability=True, random_state=42)
        best_model.fit(X_train_tfidf, y_train)

    joblib.dump(best_model, MODELS_DIR / f'{name}.pkl')
    print(f"  -> sauvegarde dans models/{name}.pkl")
    
joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl") #sérialisation et sauvegarde du vectoriseur
print(f"Vectoriseur sauvegardé dans {MODELS_DIR / 'vectorizer.pkl'}") #affichage du chemin où le vectoriseur a été sauvegardé
print("\nTermine : 4 modèles entraînés et sauvegardés avec succès.")
 