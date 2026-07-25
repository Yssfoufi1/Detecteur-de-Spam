from pathlib import Path
import joblib
import streamlit as st
from src.preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
MODEL_NAMES = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']

@st.cache_resource
def load_models():
    vectorizer = joblib.load(MODELS_DIR / 'vectorizer.pkl')
    models = {name: joblib.load(MODELS_DIR / f'{name}.pkl') for name in MODEL_NAMES}
    return vectorizer, models

vectorizer, models = load_models()

DISPLAY_NAMES = {
    'naive_bayes': 'Naive Bayes',
    'logistic_regression': 'Logistic Regression',
    'svm': 'SVM',
    'random_forest': 'Random Forest',
}
BEST_MODEL = 'logistic_regression'  # determine par evaluate.py (meilleur F1-score)

st.title("📧Detecteur de Spam")
st.write("Entrez un message pour verifier s'il s'agit de spam ou non.")

email_text = st.text_area("Message a verifier", height=150)

if st.button("Verifier"):
    if email_text.strip() == "":
        st.warning("Merci d'entrer un message.")
    else:
        cleaned = clean_text(email_text)
        vect = vectorizer.transform([cleaned])

        results = {}
        for name, model in models.items():
            pred = model.predict(vect)[0]
            proba = model.predict_proba(vect)[0]
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

        st.divider()
        st.subheader(f"Verdict final ({DISPLAY_NAMES[BEST_MODEL]} — meilleur F1-score)")
        spam_votes = sum(results[name]['pred'] for name in MODEL_NAMES)
        ham_votes = len(MODEL_NAMES) - spam_votes

        if spam_votes > ham_votes:
            final_pred = 1
        elif ham_votes > spam_votes:
            final_pred = 0
        else:
            final_pred = results[BEST_MODEL]['pred']  # egalite 2-2 : tranche par le meilleur modele (F1)

        st.divider()
        st.subheader("Verdict final (vote majoritaire)")
        if final_pred == 1:
            st.error(f"### SPAM  —  {spam_votes}/4 modeles d'accord")
        else:
            st.success(f"### Pas spam  —  {ham_votes}/4 modeles d'accord")