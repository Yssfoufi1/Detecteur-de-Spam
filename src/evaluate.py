from pathlib import Path
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
from preprocessing import load_and_prepare_data, extract_extra_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'spam_with_phishing.csv'
MODELS_DIR = BASE_DIR / 'models'

# Chargement des donnees + features (identique a train_models.py)
df = load_and_prepare_data(DATA_PATH)
X_text = df['clean_text']
X_extra = extract_extra_features(df['text'])
y = df['target']

# Meme split (meme random_state) -> reproduit exactement le meme X_test qu'a l'entrainement
X_text_train, X_text_test, X_extra_train, X_extra_test, y_train, y_test = train_test_split(
    X_text, X_extra, y, test_size=0.2, random_state=42, stratify=y
)

# Chargement du vectorizer et du scaler DEJA entraines (pas de fit ici)
vectorizer = joblib.load(MODELS_DIR / 'vectorizer.pkl')
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')

# Transformation du jeu de test (transform seulement, jamais fit_transform)
X_test_tfidf = vectorizer.transform(X_text_test)
X_extra_test_scaled = scaler.transform(X_extra_test)
X_test_combined = hstack([X_test_tfidf, X_extra_test_scaled])

# Evaluation de chaque modele sur le jeu de test
model_names = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']
results = {}
for name in model_names:
    model = joblib.load(MODELS_DIR / f'{name}.pkl')
    y_pred = model.predict(X_test_combined)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'recall': recall, 'precision': precision, 'f1_score': f1}
    print(f"\n{'='*50}\nModele : {name}\n{'='*50}")
    print(f"Recall (spam) : {recall:.4f} | Precision : {precision:.4f} | F1 : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam'], zero_division=0))

# Selection du meilleur modele : recall en priorite, precision comme critere de depart en cas d'egalite
best_model_name = max(results, key=lambda name: (results[name]['recall'], results[name]['precision']))
print(f"\nMEILLEUR MODELE (recall) : {best_model_name} (recall = {results[best_model_name]['recall']:.4f})")

# Exploration du seuil de decision sur le meilleur modele
best_model = joblib.load(MODELS_DIR / f'{best_model_name}.pkl')
proba = best_model.predict_proba(X_test_combined)[:, 1]
print(f"\nEffet du seuil de decision sur {best_model_name} :")
print(f"{'Seuil':>6} | {'Precision':>10} | {'Recall':>8} | {'F1':>6}")
for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
    y_pred_t = (proba >= threshold).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f1_t = f1_score(y_test, y_pred_t, zero_division=0)
    print(f"{threshold:>6.1f} | {p:>10.3f} | {r:>8.3f} | {f1_t:>6.3f}")