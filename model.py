# model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Données factices (Mock Data) pour tester de façon isolée
mock_texts = [
    "Félicitations ! Vous avez gagné un iPhone, cliquez ici : http://bit.ly/xxx",
    "Salut, on se voit à quelle heure pour étudier la leçon ?"
]
mock_labels = [1, 0] # 1 = Spam, 0 = Sain

# 2. Initialisation et entraînement factice
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(mock_texts)

model = MultinomialNB()
model.fit(X_train, mock_labels)

# 3. Fonction exposée pour l'interface utilisateur
def predict_spam(text: str) -> dict:
    """
    Prend en entrée un texte brut ou nettoyé, applique la vectorisation 
    et le modèle entraîné.
    """
    # Transformer le texte reçu
    X_new = vectorizer.transform([text])
    
    # Faire la prédiction
    prediction = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0].max() * 100
    
    # Renvoyer le dictionnaire exact attendu par app.py
    return {
        "is_spam": bool(prediction == 1),
        "probability": round(float(prob), 2),
        "label_text": "Spam / Phishing" if prediction == 1 else "Message Sain"
    }

# Petit test local (à effacer plus tard)
if __name__ == "__main__":
    print(predict_spam("Urgent, cliquez ici pour votre cadeau !"))