from pathlib import Path #importation de la bibliothèque pathlib pour gérer les chemins de fichiers
import joblib #importation de la bibliothèque joblib pour la sérialisation des modèles
from sklearn.model_selection import train_test_split #importation de la fonction train_test_split pour diviser les données en ensembles d'entraînement et de test
from sklearn.metrics import classification_report, f1_score, accuracy_score #importation des fonctions pour évaluer les performances des modèles
from preprocessing import load_and_prepare_data #importation de la fonction load_and_prepare_data pour charger et préparer les données

BASE_DIR = Path(__file__).resolve().parent.parent #définition du répertoire de base du projet
DATA_PATH = BASE_DIR / 'data' / 'spam.csv' #définition du chemin vers le fichier de données
MODELS_DIR = BASE_DIR / 'models' #définition du répertoire pour enregistrer les modèles

df = load_and_prepare_data(DATA_PATH) #chargement et préparation des données
X = df['clean_text'] #sélection des textes nettoyés comme caractéristiques
y = df['target'] #sélection des étiquettes comme cibles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #division des données en ensembles d'entraînement et de test
vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl") #chargement du vectoriseur sauvegardé
X_test_tfidf = vectorizer.transform(X_test) #vectorisation des textes de test

model_name = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest'] #liste des noms des modèles à évaluer
results = {} #dictionnaire pour stocker les résultats des modèles
for name in model_name:
    model = joblib.load(MODELS_DIR / f"{name}.pkl") #chargement du modèle sauvegardé
    y_pred = model.predict(X_test_tfidf) #prédiction des étiquettes pour les textes de test
    accuracy = accuracy_score(y_test, y_pred) #calcul de la précision du modèle
    f1 = f1_score(y_test, y_pred) #calcul du score F1 du modèle
    report = classification_report(y_test, y_pred) #génération du rapport de classification
    results[name] = {'accuracy': accuracy, 'f1_score': f1, 'classification_report': report} #stockage des résultats dans le dictionnaire

    print(f"Résultats pour le modèle {name}:") #affichage du nom du modèle
    print("Rapport de classification:") #affichage du rapport de classification
    print(report) #affichage du rapport de classification
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam'])) #affichage du rapport de classification avec les noms des classes

best_model_name = max(results, key=lambda name: results[name]['f1_score']) #sélection du modèle avec le meilleur score F1
best_f1_score = results[best_model_name] #récupération du score F1 du meilleur modèle
print(f"Le meilleur modèle est {best_model_name} avec un score F1 de {best_f1_score['f1_score']:.4f}") #affichage du meilleur modèle et de son score F1