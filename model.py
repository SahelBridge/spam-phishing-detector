"""
model.py - Module de détection de Spam & Phishing
Entraîné sur dataset_clean.csv (et compatible dataset_clean.csv.gz).
Fournit la fonction `predict_spam(text)` compatible avec l'interface utilisateur Streamlit / API.
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Chemins des fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(BASE_DIR, "dataset_clean.csv")
DATASET_GZ = os.path.join(BASE_DIR, "dataset_clean.csv.gz")
DATASET_TEST = os.path.join(BASE_DIR, "dataset_clean_test.csv")
MODEL_FILE = os.path.join(BASE_DIR, "spam_detector_model.joblib")


def find_dataset_path(custom_path: str = None) -> str:
    """Trouve le chemin du dataset disponible."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if os.path.exists(DATASET_CSV):
        return DATASET_CSV
    if os.path.exists(DATASET_GZ):
        return DATASET_GZ
    if os.path.exists(DATASET_TEST):
        return DATASET_TEST
    raise FileNotFoundError(
        "Aucun fichier de dataset trouvé (cherché : dataset_clean.csv, dataset_clean.csv.gz, dataset_clean_test.csv)."
    )


def train_model(dataset_path: str = None, save_model: bool = True) -> Pipeline:
    """
    Entraîne le modèle TF-IDF + MultinomialNB sur les données nettoyées
    et sauvegarde le pipeline dans spam_detector_model.joblib.
    """
    path = find_dataset_path(dataset_path)
    print(f"🔄 Chargement des données depuis : {path} ...")

    # Lecture du dataset (pandas gère automatiquement la décompression .gz)
    df = pd.read_csv(path)

    # Nettoyage de base : suppression des valeurs nulles
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    print(f"📊 Données chargées : {len(df)} exemples.")
    print(f"📈 Répartition des classes : {df['label'].value_counts().to_dict()} (0=Sain, 1=Spam)")

    # Création du pipeline ML
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents=None
            )
        ),
        ("classifier", MultinomialNB(alpha=0.1))
    ])

    print("⚙️ Entraînement du modèle en cours...")
    pipeline.fit(df["text"], df["label"])
    print("✅ Entraînement terminé avec succès !")

    if save_model:
        joblib.dump(pipeline, MODEL_FILE, compress=3)
        print(f"💾 Modèle sauvegardé dans : {MODEL_FILE}")

    return pipeline


def load_or_train_pipeline() -> Pipeline:
    """
    Charge le pipeline pré-entraîné s'il existe, sinon l'entraîne automatiquement.
    """
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du modèle ({e}), ré-entraînement en cours...")

    return train_model(save_model=True)


# Initialisation globale du pipeline (chargé en mémoire au premier import)
_pipeline = None


def get_pipeline() -> Pipeline:
    """Retourne l'instance du pipeline chargé (singleton)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = load_or_train_pipeline()
    return _pipeline


def predict_spam(text: str) -> dict:
    """
    Prend en entrée un texte brut ou nettoyé, applique le pipeline
    (vectorisation TF-IDF + modèle MultinomialNB) et renvoie le dictionnaire
    de résultat attendu par l'interface utilisateur.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "is_spam": False,
            "probability": 0.0,
            "label_text": "Message Vide"
        }

    pipeline = get_pipeline()

    prediction = int(pipeline.predict([text])[0])
    probabilities = pipeline.predict_proba([text])[0]

    # Indice 0 = Sain, Indice 1 = Spam
    confidence = float(probabilities[prediction]) * 100.0

    return {
        "is_spam": bool(prediction == 1),
        "probability": round(confidence, 2),
        "label_text": "Spam / Phishing" if prediction == 1 else "Message Sain"
    }


if __name__ == "__main__":
    # Si exécuté avec argument --train ou si le modèle n'a pas encore été généré
    if "--train" in sys.argv or not os.path.exists(MODEL_FILE):
        print("🚀 Lancement de l'entraînement...")
        train_model()

    # Tests d'exemples
    exemples = [
        "Félicitations ! Vous avez gagné un iPhone, cliquez ici : http://bit.ly/xxx",
        "Salut, on se voit à quelle heure pour étudier la leçon ?",
        "URGENT: Votre compte bancaire est temporairement suspendu. Cliquez ici http://bank-security.com",
        "N'oublie pas d'envoyer le compte-rendu de la réunion avant demain midi merci."
    ]

    print("\n🔍 --- TESTS DE PRÉDICTION ---")
    for ex in exemples:
        res = predict_spam(ex)
        print(f"\nTexte : {ex}")
        print(f"Résultat : {res}")