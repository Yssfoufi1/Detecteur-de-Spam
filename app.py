from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from src.preprocessing import clean_text, extract_extra_features

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
MODEL_NAMES = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']
DISPLAY_NAMES = {
    'naive_bayes': 'Naive Bayes',
    'logistic_regression': 'Logistic Regression',
    'svm': 'SVM',
    'random_forest': 'Random Forest',
}
BEST_MODEL = 'random_forest'
RECALL_THRESHOLD = 0.4  # badal 0.3 -- recall qriب mn maximum (0.986) bla ma nkhesro precision bzaf

@st.cache_resource
def load_models():
    vectorizer = joblib.load(MODELS_DIR / 'vectorizer.pkl')
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    models = {name: joblib.load(MODELS_DIR / f'{name}.pkl') for name in MODEL_NAMES}
    return vectorizer, scaler, models

vectorizer, scaler, models = load_models()

st.title("Detecteur de Spam")
st.write("Entrez un message pour verifier s'il s'agit de spam ou non.")

email_text = st.text_area("Message a verifier", height=150)

if st.button("Verifier"):
    if email_text.strip() == "":
        st.warning("Merci d'entrer un message.")
    else:
        cleaned = clean_text(email_text)
        vect_tfidf = vectorizer.transform([cleaned])
        extra = extract_extra_features(pd.Series([email_text]))
        extra_scaled = scaler.transform(extra)
        vect_combined = hstack([vect_tfidf, extra_scaled])

        results = {}
        for name, model in models.items():
            pred = model.predict(vect_combined)[0]
            proba = model.predict_proba(vect_combined)[0]
            results[name] = {
                'pred': pred,
                'ham_pct': proba[0] * 100,
                'spam_pct': proba[1] * 100,
            }

        st.subheader("Resultats par algorithme")
        cols = st.columns(4)
        for col, name in zip(cols, MODEL_NAMES):
            res = results[name]
            with col:
                st.markdown(f"**{DISPLAY_NAMES[name]}**")
                if res['pred'] == 1:
                    st.error("SPAM")
                else:
                    st.success("HAM")
                st.write(f"Spam : {res['spam_pct']:.1f}%")
                st.write(f"Ham : {res['ham_pct']:.1f}%")

        spam_votes = sum(results[name]['pred'] for name in MODEL_NAMES)
        avg_spam_pct = sum(results[name]['spam_pct'] for name in MODEL_NAMES) / len(MODEL_NAMES)

        best_spam_proba = results[BEST_MODEL]['spam_pct'] / 100
        final_pred = 1 if best_spam_proba >= RECALL_THRESHOLD else 0

        st.divider()
        st.subheader(f"Verdict final ({DISPLAY_NAMES[BEST_MODEL]}, seuil = {RECALL_THRESHOLD})")
        if final_pred == 1:
            st.error("### SPAM")
        else:
            st.success("### Pas spam")
        st.write(f"Probabilite spam : {results[BEST_MODEL]['spam_pct']:.1f}% (seuil : {RECALL_THRESHOLD*100:.0f}%, optimise pour recall maximal)")
        st.caption(f"Vote majoritaire (info) : {spam_votes}/4 modeles disent spam, moyenne = {avg_spam_pct:.1f}%")