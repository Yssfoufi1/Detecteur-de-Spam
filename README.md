# Spam Detector — Détection de Spam par Machine Learning

## Description

Ce projet a pour objectif de détecter automatiquement si un message (SMS/email) est un spam ou non, en comparant les performances de quatre algorithmes de classification supervisée : Naive Bayes, Logistic Regression, SVM et Random Forest.

## Architecture du projet
spam-detector/
├── data/
│   └── spam_dataset.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluate.py
├── models/
│   ├── vectorizer.pkl
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── random_forest.pkl
├── app.py
├── requirements.txt
└── README.md

## Installation

```bash
pip install -r requirements.txt
```

Les ressources NLTK nécessaires (wordnet, punkt, averaged_perceptron_tagger) sont téléchargées automatiquement au premier lancement du script.

## Utilisation

**1. Exploration des données** — ouvrir `notebooks/exploration.ipynb` (distribution des classes, longueur des messages, mots fréquents).

**2. Entraînement des modèles**
```bash
python src/train_models.py
```
Charge les données, effectue un `GridSearchCV` (validation croisée à 5 plis) pour optimiser les hyperparamètres de chaque modèle, puis sauvegarde les 4 modèles et le vectoriseur TF-IDF dans `models/`.

**3. Évaluation**
```bash
python src/evaluate.py
```
Évalue les 4 modèles sur le jeu de test (20%, non vu pendant l'entraînement) et sélectionne automatiquement le meilleur modèle selon le F1-score de la classe spam.

**4. Interface utilisateur**
```bash
streamlit run app.py
```
Tableau de bord interactif : saisie d'un message, verdict et pourcentage de confiance de chacun des 4 modèles, puis verdict final du meilleur modèle.

## Méthodologie

**Prétraitement** : mise en minuscule, suppression de la ponctuation, tokenisation, lemmatisation avec étiquetage grammatical (POS tagging) pour ramener les verbes à leur forme infinitive.

**Vectorisation** : TF-IDF, avec suppression des mots vides (stop words) anglais.

**Déséquilibre des classes** (≈87% ham / 13% spam) : géré par une répartition stratifiée (`stratify=y`) lors du split train/test, et par `class_weight='balanced'` pour Logistic Regression, SVM et Random Forest.

**Optimisation des hyperparamètres** : `GridSearchCV`, validation croisée à 5 plis, optimisée sur le F1-score de la classe spam.

## Résultats

| Modèle | Précision (spam) | Rappel (spam) | F1-score (spam) |
|---|---|---|---|
| Naive Bayes | 0.97 | 0.85 | 0.91 |
| Logistic Regression | 0.95 | 0.91 | **0.93** |
| SVM | 0.97 | 0.89 | 0.92 |
| Random Forest | 0.97 | 0.85 | 0.91 |

**Meilleur modèle : Logistic Regression** (F1-score = 0.9297), sélectionné automatiquement par `evaluate.py`.

## Limites et perspectives

Le jeu de données (SMS Spam Collection Dataset) date d'environ 2011-2012 et reflète des schémas de spam de cette époque (jeux-concours, numéros surtaxés). Des tests informels montrent que les modèles généralisent moins bien à des formes de spam plus récentes, comme le phishing bancaire, peu représentées dans le vocabulaire d'entraînement. Une amélioration future pourrait consister à enrichir le jeu de données avec des exemples plus contemporains.

## Auteur

Youssef Aoufi
ESTF — Data Engineering