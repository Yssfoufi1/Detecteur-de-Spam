# Importation des modules necessaires
from pathlib import Path
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_prepare_data, extract_extra_features

# Chemins bases sur l'emplacement du fichier (fonctionne peu importe le dossier de lancement)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'spam_with_phishing.csv'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Chargement des donnees + extraction des features (texte nettoye + features numeriques)
df = load_and_prepare_data(DATA_PATH)
X_text = df['clean_text']
X_extra = extract_extra_features(df['text'])
y = df['target']

# Split train/test (80/20, stratifie pour garder la meme proportion spam/ham)
X_text_train, X_text_test, X_extra_train, X_extra_test, y_train, y_test = train_test_split(
    X_text, X_extra, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorisation TF-IDF : fit uniquement sur train, transform sur test
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_test_tfidf = vectorizer.transform(X_text_test)

# Mise a l'echelle des features numeriques (0-1, meme echelle que le TF-IDF)
scaler = MinMaxScaler()
X_extra_train_scaled = scaler.fit_transform(X_extra_train)
X_extra_test_scaled = scaler.transform(X_extra_test)

# Combinaison du TF-IDF (sparse) et des features numeriques en une seule matrice
X_train_combined = hstack([X_train_tfidf, X_extra_train_scaled])
X_test_combined = hstack([X_test_tfidf, X_extra_test_scaled])

# Grilles d'hyperparametres a tester pour chaque modele
param_grids = {
    'naive_bayes': {'alpha': [0.1, 0.5, 1.0, 2.0]},
    'logistic_regression': {'C': [0.1, 1, 10]},
    'svm': {'C': [0.1, 1, 10]},
    'random_forest': {'n_estimators': [100, 200], 'max_depth': [None, 30]},
}

# Modeles de base (class_weight='balanced' pour compenser le desequilibre ham/spam)
base_models = {
    'naive_bayes': MultinomialNB(),
    'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'svm': SVC(kernel='linear', class_weight='balanced', random_state=42),
    'random_forest': RandomForestClassifier(class_weight='balanced', random_state=42),
}

# Entrainement : GridSearchCV (validation croisee 5 plis, optimise sur le recall)
for name, model in base_models.items():
    print(f"\nGridSearch : {name} ...")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='recall', n_jobs=-1)
    grid.fit(X_train_combined, y_train)
    print(f"  Meilleurs parametres : {grid.best_params_}")
    print(f"  Meilleur recall (validation croisee) : {grid.best_score_:.4f}")
    best_model = grid.best_estimator_

    # Cas particulier SVM : probability=True est lent en GridSearch, on reentraine
    # une seule fois avec le meilleur C trouve, pour avoir les scores de confiance
    if name == 'svm':
        best_C = grid.best_params_['C']
        best_model = SVC(kernel='linear', C=best_C, class_weight='balanced', probability=True, random_state=42)
        best_model.fit(X_train_combined, y_train)

    joblib.dump(best_model, MODELS_DIR / f'{name}.pkl')

# Sauvegarde du vectorizer et du scaler (necessaires pour reutiliser les modeles)
joblib.dump(vectorizer, MODELS_DIR / 'vectorizer.pkl')
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
print("\nTermine.")