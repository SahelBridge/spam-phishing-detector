import streamlit as st
from model import predict_spam

# Configuration
st.set_page_config(
    page_title="Détecteur de Spam ",
    page_icon="🛡️",
    layout="centered"
)

# 2. CSS 

st.markdown("""
    <style>
    /* Style global pour les conteneurs (cartes) */
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="ststContainer"]) {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); /* Ombre douce */
        margin-bottom: 20px;
    }
    
    /* Personnalisation du bouton principal (Bleu Moderne) */
    div.stButton > button {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #0056b3 !important; /* Bleu plus foncé au survol */
        box-shadow: 0 2px 8px rgba(0,123,255,0.3);
    }
    </style>
""", unsafe_allow_html=True)


# 3. EN-TÊTE 
st.markdown("<h1 style='text-align: center; color: #333;'>🛡️ Analyseur de Messages Intelligent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1em;'>Protégez-vous des spams et du phishing en un clic.</p>", unsafe_allow_html=True)
st.write("") # Espace


# 4. ZONE DE SAISIE 
with st.container():
    st.markdown("<h3 style='color: #007bff; margin-bottom: 15px;'>📩 Collez votre message</h3>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Le texte apparaîtra ici :",
        height=150,
        placeholder="Exemple : Félicitations ! Vous avez gagné un cadeau, cliquez ici..."
    )
    st.write("") # Espace avant bouton
    analyze_button = st.button("🔍 Analyser maintenant", use_container_width=True)


# 5. ZONE DE RÉSULTATS 
if analyze_button:
    st.write("")
    if not user_input.strip():
        st.warning("⚠️ Veuillez d'abord saisir un message.")
    else:
        # Analyse via le modèle ML entraîné
        result = predict_spam(user_input)
        is_spam = result["is_spam"]
        probability = result["probability"]
        
        with st.container():
            st.markdown("<h3 style='color: #333;'>📊 Résultat de l'analyse</h3>", unsafe_allow_html=True)
            st.write("")
            
            if is_spam:
                # Carte ROUGE pour Spam
                st.markdown("""
                    <div style='background-color: #ffebee; border-left: 5px solid #ef5350; padding: 15px; border-radius: 8px; color: #c62828;'>
                        <strong>🚨 Alerte : Message très suspect !</strong><br>
                        Ce message présente de fortes caractéristiques de Spam ou de Phishing. Soyez prudent.
                    </div>
                """, unsafe_allow_html=True)
                st.write("")
                st.metric(
                    label="Niveau de certitude (Spam / Phishing)",
                    value=f"{probability} %",
                    delta="Risque Élevé",
                    delta_color="inverse"
                )
                
            else:
                # Carte VERTE pour Sûr
                st.markdown("""
                    <div style='background-color: #e8f5e9; border-left: 5px solid #66bb6a; padding: 15px; border-radius: 8px; color: #2e7d32;'>
                        <strong>✅ Message légitime : Aucun risque détecté.</strong><br>
                        Ce message semble sûr.
                    </div>
                """, unsafe_allow_html=True)
                st.write("")
                st.metric(
                    label="Niveau de certitude (Message Sain)",
                    value=f"{probability} %",
                    delta="Risque Faible",
                    delta_color="normal"
                )