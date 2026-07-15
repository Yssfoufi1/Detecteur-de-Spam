from pathlib import Path
import sys
import joblib
import streamlit as st

# =========================
# Configuration
# =========================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocessing import clean_text

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered"
)


# =========================
# Load Model
# =========================
@st.cache_resource
def load_artifacts():
    vectorizer_path = BASE_DIR / "models" / "vectorizer.pkl"
    model_path = BASE_DIR / "models" / "svm.pkl"

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer introuvable : {vectorizer_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    return vectorizer, model


# =========================
# Prediction
# =========================
def predict_message(message):
    vectorizer, model = load_artifacts()

    cleaned = clean_text(message)

    features = vectorizer.transform([cleaned])

    prediction = int(model.predict(features)[0])

    label = "Spam" if prediction == 1 else "Ham"

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction] * 100

    return label, confidence, cleaned


# =========================
# UI
# =========================
st.title("📧 Spam Detector")

st.write(
    "Entrez un message puis cliquez sur **Prédire** pour savoir s'il s'agit d'un spam."
)

message = st.text_area(
    "Message",
    height=180,
    placeholder="Congratulations! You won a free iPhone. Click here..."
)

if st.button("Prédire"):

    if not message.strip():
        st.warning("Veuillez entrer un message.")
        st.stop()

    try:
        label, confidence, cleaned = predict_message(message)

        if label == "Spam":
            st.error(f"🚨 Résultat : {label}")
        else:
            st.success(f"✅ Résultat : {label}")

        if confidence is not None:
            st.metric("Confiance", f"{confidence:.2f}%")

        with st.expander("Texte après prétraitement"):
            st.write(cleaned)

    except Exception as e:
        st.error(f"Erreur : {e}")