# Détecteur de Spam & Phishing — Machine Learning

Systeme de detection automatique de spam et de tentatives de phishing dans les emails,
combinant TF-IDF, feature engineering cible et quatre algorithmes de classification.

## Architecture
Detecteur-de-Spam/
├── data/
│   ├── spam.csv                   # SMS Spam Collection Dataset (source initiale)
│   ├── Nazario.csv                # Nazario Phishing Corpus (enrichissement)
│   └── spam_with_phishing.csv     # Dataset fusionne (source d'entrainement finale)
├── notebooks/
│   └── exploration.ipynb          # Analyse exploratoire des donnees (EDA)
├── src/
│   ├── preprocessing.py           # Nettoyage, lemmatisation, feature engineering
│   ├── train_models.py            # Entrainement + GridSearchCV des 4 modeles
│   └── evaluate.py                # Evaluation + selection du meilleur modele
├── models/                        # Modeles entraines (.pkl), generes par train_models.py
├── app.py                         # Interface Streamlit
├── requirements.txt
└── README.md
## Installation

```bash
pip install -r requirements.txt
```

Les ressources NLTK (tokenization, POS tagging, lemmatisation) sont telechargees
automatiquement au premier lancement.

## Utilisation

**1. Analyse exploratoire** — ouvrir `notebooks/exploration.ipynb`

**2. Entrainement**
```bash
python src/train_models.py
```
Charge `spam_with_phishing.csv`, extrait TF-IDF + 6 features numeriques, optimise
les hyperparametres de chaque modele par GridSearchCV (scoring='recall', cv=5),
sauvegarde les modeles dans `models/`.

**3. Evaluation**
```bash
python src/evaluate.py
```
Evalue les 4 modeles sur le jeu de test, selectionne le meilleur (recall puis
precision), affiche l'effet du seuil de decision.

**4. Application**
```bash
streamlit run app.py
```
Interface web : saisie d'un message, verdict de chacun des 4 modeles + verdict
final officiel.

**Demo en ligne :** https://spam-detecteur.streamlit.app/

## Methodologie

- **Pretraitement** : minuscule, suppression ponctuation, tokenisation, lemmatisation
  avec POS tagging (NLTK) — les verbes sont ramenes a leur forme infinitive.
- **Feature engineering** : en plus du TF-IDF, 6 features numeriques par message —
  longueur, ratio de majuscules, nombre de `!`, nombre de chiffres, presence d'URL,
  et un score de mots-cles typiques du phishing (`reset your password`,
  `verify your account`, etc.).
- **Desequilibre des classes** : gere par `stratify` au split et `class_weight='balanced'`.
- **Optimisation** : GridSearchCV (5-fold), optimise sur le **recall** (priorite :
  capturer un maximum de spam), puis seuil de decision ajuste a 0.4.
- **Jeu de donnees enrichi** : fusion de SMS Spam Collection (5169 messages) avec
  le Nazario Phishing Corpus (1565 emails de phishing reels), pour ameliorer la
  detection du phishing moderne, absent du dataset original (2011-2012).

## Resultats (jeu de test, 1347 messages)

| Modele               | Recall | Precision | F1     |
|-----------------------|--------|-----------|--------|
| Naive Bayes            | 0.932  | 0.954     | 0.943  |
| Logistic Regression    | 0.946  | 0.988     | 0.967  |
| SVM                     | 0.955  | 0.986     | 0.970  |
| **Random Forest** (retenu) | **0.978**  | **0.995**     | **0.986**  |

## Limites

- Optimise principalement pour l'anglais.
- Approche bag-of-words (TF-IDF) : moins efficace face a un phishing tres bien
  redige, sans marqueurs lexicaux evidents.

## Auteur

Youssef Aoufi — Data engineering