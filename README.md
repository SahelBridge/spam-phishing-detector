# 🛡️ Spam & Phishing Detector

Application intelligente d'analyse et de détection de **Spams** et de tentatives de **Phishing** utilisant le Traitement Automatique du Langage Naturel (NLP) et le Machine Learning, dotée d'une interface web moderne et interactive développée avec **Streamlit**.

---

## 🌟 Fonctionnalités

- **Détection en temps réel** : Analyse instantanée de messages suspects (emails, SMS, alertes bancaires, faux colis, etc.).
- **Modèle Machine Learning Performant** :
  - Extraction de caractéristiques textuelles avec **TF-IDF** (uni-grammes et bi-grammes, pondération sublinéaire, 50 000 termes).
  - Classifieur **Multinomial Naive Bayes** entraîné sur plus de **92 000 exemples** équilibrés.
  - Précision globale supérieure à **96.6%**.
- **Persistance optimisée (`joblib`)** : Modèle sérialisé prêt à l'emploi (`spam_detector_model.joblib`) garantissant un démarrage de l'application en moins de 0.2 seconde.
- **Interface Utilisateur Moderne** : Dashboard Streamlit ergonomique avec affichage du niveau de risque et de la probabilité de confiance.

---

## 📁 Architecture du Projet

```text
spam-phishing-detector/
├── app.py                     # Interface utilisateur interactive Streamlit
├── model.py                   # Pipeline NLP (TF-IDF + Naive Bayes), entraînement et fonction d'inférence
├── clean_data.py              # Script de fusion et de nettoyage des datasets sources
├── spam_detector_model.joblib # Modèle pré-entraîné sérialisé
├── dataset_clean.csv.gz       # Dataset nettoyé compressé (~92k messages)
├── dataset_clean_test.csv     # Échantillons de test rapides
├── requirements.txt           # Dépendances Python du projet
└── data/                      # Jeux de données sources (Enron, SpamAssassin, Nazario, etc.)
```

---

## 🚀 Installation & Démarrage Rapide

### 1. Cloner le projet
```bash
git clone https://github.com/SahelBridge/spam-phishing-detector.git
cd spam-phishing-detector
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application Web
```bash
streamlit run app.py
```
L'interface sera accessible directement dans votre navigateur sur `http://localhost:8501`.

---

## 🤖 Utilisation en Ligne de Commande

### Tester le modèle directement
```bash
python model.py
```

### Forcer un ré-entraînement sur le dataset
```bash
python model.py --train
```

### Intégration en tant que module Python
```python
from model import predict_spam

result = predict_spam("URGENT: Your bank account is locked. Verify at http://scam.link")
print(result)
# Sortie : {'is_spam': True, 'probability': 99.84, 'label_text': 'Spam / Phishing'}
```

---

## 📦 Dataset

Le projet inclut le jeu de données nettoyé et compressé (`dataset_clean.csv.gz`). 

Si vous souhaitez télécharger la version brute décompressée (`dataset_clean.csv`) :
👉 [Télécharger le Dataset sur Google Drive](https://drive.google.com/file/d/1S_nlTWeyF3Sj0rAZudmVKuIoQSYkLOHQ/view?usp=sharing)

Placez le fichier à la racine du projet si vous souhaitez l'utiliser sans décompression.

---

## 👥 Auteurs & Organisation

Projet développé par **SahelBridge / Farafina Lab**.

