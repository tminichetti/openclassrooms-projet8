"""
Application Streamlit pour la démonstration du système de segmentation d'images
Projet: Véhicules Autonomes - Future Vision Transport
"""

import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import json
import os

# Configuration de la page
st.set_page_config(
    page_title="Segmentation d'Images - Véhicules Autonomes",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API (peut être modifiée via variable d'environnement)
API_URL = os.environ.get('API_URL', 'http://localhost:5000')

# Palette de couleurs pour les 8 classes Cityscapes
CITYSCAPES_COLORS = {
    0: [128, 64, 128],   # Void / Background
    1: [244, 35, 232],   # Flat
    2: [70, 70, 70],     # Construction
    3: [102, 102, 156],  # Object
    4: [190, 153, 153],  # Nature
    5: [153, 153, 153],  # Sky
    6: [250, 170, 30],   # Human
    7: [220, 220, 0],    # Vehicle
}

CITYSCAPES_LABELS = {
    0: "Void/Background",
    1: "Flat",
    2: "Construction",
    3: "Object",
    4: "Nature",
    5: "Sky",
    6: "Human",
    7: "Vehicle"
}

def mask_to_rgb(mask):
    """
    Convertit un mask de classes en image RGB colorée

    Args:
        mask: Array 2D avec les indices de classes

    Returns:
        Image RGB avec les couleurs correspondantes
    """
    height, width = len(mask), len(mask[0]) if mask else 0
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in CITYSCAPES_COLORS.items():
        for i in range(height):
            for j in range(width):
                if mask[i][j] == class_id:
                    rgb_mask[i, j] = color

    return rgb_mask

def check_api_health():
    """Vérifie si l'API est accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('status') == 'healthy', data.get('model_loaded', False)
        return False, False
    except:
        return False, False

def predict_segmentation(image):
    """
    Envoie une image à l'API et récupère la prédiction

    Args:
        image: Image PIL

    Returns:
        dict avec le mask et les informations
    """
    # Convertir l'image en bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Envoyer la requête
    files = {'image': ('image.png', img_byte_arr, 'image/png')}

    try:
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

        if response.status_code == 200:
            return response.json(), None
        else:
            error_data = response.json()
            return None, f"Erreur API: {error_data.get('message', 'Erreur inconnue')}"

    except requests.exceptions.Timeout:
        return None, "Timeout: L'API a mis trop de temps à répondre"
    except requests.exceptions.ConnectionError:
        return None, "Erreur de connexion: Impossible de joindre l'API"
    except Exception as e:
        return None, f"Erreur: {str(e)}"

# Interface principale
def main():
    # Header
    st.title("🚗 Segmentation d'Images pour Véhicules Autonomes")
    st.markdown("**Future Vision Transport** - Système de Vision par Ordinateur")
    st.markdown("---")

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # URL de l'API
        api_url_input = st.text_input(
            "URL de l'API",
            value=API_URL,
            help="URL de l'API de prédiction (Heroku ou local)"
        )

        if api_url_input != API_URL:
            globals()['API_URL'] = api_url_input

        # Vérification de l'API
        st.markdown("### 🔍 État de l'API")
        if st.button("Vérifier l'API", use_container_width=True):
            with st.spinner("Vérification..."):
                is_healthy, model_loaded = check_api_health()

                if is_healthy:
                    st.success("✅ API accessible")
                    if model_loaded:
                        st.success("✅ Modèle chargé")
                    else:
                        st.error("❌ Modèle non chargé")
                else:
                    st.error("❌ API inaccessible")

        # Informations
        st.markdown("---")
        st.markdown("### 📊 Classes de Segmentation")
        for class_id, label in CITYSCAPES_LABELS.items():
            color = CITYSCAPES_COLORS[class_id]
            st.markdown(
                f"<div style='display: flex; align-items: center;'>"
                f"<div style='width: 20px; height: 20px; background-color: rgb({color[0]}, {color[1]}, {color[2]}); "
                f"margin-right: 10px; border: 1px solid #ccc;'></div>"
                f"<span>{label}</span></div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### ℹ️ À propos")
        st.info(
            "Cette application permet de visualiser les résultats de segmentation "
            "d'images pour le système de vision embarqué des véhicules autonomes."
        )

    # Zone principale
    col1, col2 = st.columns(2)

    with col1:
        st.header("📤 Image d'entrée")

        # Upload d'image
        uploaded_file = st.file_uploader(
            "Choisissez une image",
            type=['png', 'jpg', 'jpeg'],
            help="Téléchargez une image de scène urbaine"
        )

        if uploaded_file is not None:
            # Afficher l'image originale
            image = Image.open(uploaded_file)
            st.image(image, caption="Image originale", use_container_width=True)

            # Informations sur l'image
            st.markdown(f"**Dimensions**: {image.size[0]} x {image.size[1]} pixels")
            st.markdown(f"**Format**: {image.format}")
            st.markdown(f"**Mode**: {image.mode}")

    with col2:
        st.header("🎯 Prédiction de Segmentation")

        if uploaded_file is not None:
            # Bouton de prédiction
            if st.button("🚀 Lancer la segmentation", type="primary", use_container_width=True):
                with st.spinner("Segmentation en cours..."):
                    result, error = predict_segmentation(image)

                    if error:
                        st.error(f"❌ {error}")
                    else:
                        # Stocker le résultat dans la session
                        st.session_state['result'] = result
                        st.success("✅ Segmentation réussie !")

            # Afficher le résultat s'il existe
            if 'result' in st.session_state:
                result = st.session_state['result']

                # Convertir le mask en image RGB
                mask = result.get('mask', [])
                if mask:
                    rgb_mask = mask_to_rgb(mask)
                    mask_image = Image.fromarray(rgb_mask)

                    st.image(mask_image, caption="Mask de segmentation prédit", use_container_width=True)

                    # Informations sur le résultat
                    shape = result.get('shape', {})
                    st.markdown(f"**Dimensions mask**: {shape.get('width')} x {shape.get('height')} pixels")

                    # Distribution des classes
                    st.markdown("### 📊 Distribution des classes")

                    # Calculer la distribution
                    flat_mask = [pixel for row in mask for pixel in row]
                    unique, counts = np.unique(flat_mask, return_counts=True)
                    total_pixels = len(flat_mask)

                    for class_id, count in zip(unique, counts):
                        percentage = (count / total_pixels) * 100
                        label = CITYSCAPES_LABELS.get(class_id, f"Classe {class_id}")
                        st.progress(percentage / 100, text=f"{label}: {percentage:.2f}%")

        else:
            st.info("👆 Téléchargez une image dans la colonne de gauche pour commencer")

    # Section de comparaison (si disponible)
    st.markdown("---")
    st.header("🔍 Comparaison Détaillée")

    if uploaded_file is not None and 'result' in st.session_state:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Image Originale")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### Mask Prédit")
            result = st.session_state['result']
            mask = result.get('mask', [])
            if mask:
                rgb_mask = mask_to_rgb(mask)
                mask_image = Image.fromarray(rgb_mask)
                st.image(mask_image, use_container_width=True)

        with col3:
            st.markdown("### Overlay")
            if mask:
                # Créer un overlay (50% image, 50% mask)
                img_array = np.array(image.resize((len(mask[0]), len(mask))))
                overlay = (img_array * 0.5 + rgb_mask * 0.5).astype(np.uint8)
                overlay_image = Image.fromarray(overlay)
                st.image(overlay_image, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Développé pour Future Vision Transport | Projet OpenClassrooms #8"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
