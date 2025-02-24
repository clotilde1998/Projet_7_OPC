import streamlit as st
import pandas as pd
import requests
import json
import base64
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt

# Configurer l'URL de tracking MLflow
mlflow.set_tracking_uri("http://localhost:8080")  # Assure-toi que l'URL est correcte


MODEL_URI = "models:/smote_lightgbm_pipeline_model/8"
model = mlflow.sklearn.load_model(MODEL_URI)

# Configuration de l'API
base_url = "http://127.0.0.1:8080/"
headers_request = {"Content-Type": "application/json"}

# Fonction pour obtenir les prédictions du modèle
@st.cache_data
def request_prediction(client_id):
    url_request = base_url
    data_json = {"client_id": client_id}
    response = requests.post(url_request, headers=headers_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(f"Erreur {response.status_code}: {response.text}")
    
    return response.json()["proba"], response.json()["result"]

# Fonction pour obtenir la liste des clients
@st.cache_data
def request_client_list():
    url_request = base_url
    response = requests.get(url_request, headers=headers_request)
    
    if response.status_code != 200:
        raise Exception(f"Erreur {response.status_code}: {response.text}")
    
    return [int(x) for x in response.json()["clients_list"]]

# Fonction pour l'explication globale avec SHAP
def shap_global_explanation(model, X_train):
    """
    Crée une explication globale du modèle en utilisant SHAP.
    Affiche l'importance globale des caractéristiques.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values[1], X_train, plot_type="bar")  # Résumé des valeurs SHAP pour la classe positive
    plt.title("Importance des caractéristiques (SHAP Global)")
    st.pyplot(plt)

# Fonction pour l'explication locale avec SHAP
def shap_local_explanation(model, X_train, client_id):
    """
    Expliquer une prédiction locale pour un client spécifique en utilisant SHAP.
    Affiche l'impact des caractéristiques pour la prédiction de ce client.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.initjs()  # Initialisation de la visualisation JS
    shap.force_plot(shap_values[1][client_id], X_train.iloc[client_id], feature_names=X_train.columns)
    st.pyplot(shap.force_plot(shap_values[1][client_id], X_train.iloc[client_id], feature_names=X_train.columns))

# Interface utilisateur Streamlit
st.set_page_config(page_title='Scoring de Crédit', layout='wide')
st.title("🏦 Scoring de Crédit - Prédiction d'Octroi de Prêt")

# Sidebar - Sélection du client
st.sidebar.header("📊 Sélection du client")
client_id = st.sidebar.selectbox("Sélectionner un identifiant client", request_client_list())

# Bouton de prédiction
if st.sidebar.button("🔍 Prédire"):
    proba, prediction = request_prediction(client_id)
    
    st.subheader(f"Résultat pour le client {client_id}")
    st.markdown(f"* Probabilité de remboursement: **{proba*100:.2f}%**")
    
    progress_bar = st.progress(0)
    for i in range(int(proba * 100)):
        progress_bar.progress(i + 1)
    
    if prediction == 1:
        st.error("❌ Prêt refusé !")
    else:
        st.success("✅ Prêt accordé !")

# Affichage de l'explication globale
if st.sidebar.checkbox("Afficher l'explication globale"):
    shap_global_explanation(model, pd.DataFrame())  # Remplacer pd.DataFrame() par X_train pour tes données

# Affichage de l'explication locale pour un client spécifique
if st.sidebar.button("Afficher l'explication locale"):
    shap_local_explanation(model, pd.DataFrame(), client_id)  # Remplacer pd.DataFrame() par X_train et client_id
