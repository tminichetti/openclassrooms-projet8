"""
API de prédiction pour la segmentation d'images
Système de vision pour véhicules autonomes - Future Vision Transport
"""

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io
import os

app = Flask(__name__)

# Charger le modèle au démarrage
MODEL_PATH = os.path.join('model', 'segmentation_model.h5')
model = None

def load_model():
    """Charge le modèle de segmentation"""
    global model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("✓ Modèle chargé avec succès")
    except Exception as e:
        print(f"✗ Erreur lors du chargement du modèle: {e}")
        model = None

# Charger le modèle au démarrage
load_model()

def preprocess_image(image, target_size=(256, 512)):
    """
    Prétraite une image pour la prédiction

    Args:
        image: Image PIL
        target_size: Taille cible (hauteur, largeur)

    Returns:
        Image prétraitée normalisée
    """
    # Redimensionner
    image = image.resize((target_size[1], target_size[0]))

    # Convertir en array et normaliser
    image_array = np.array(image) / 255.0

    # Ajouter dimension batch
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

def postprocess_mask(mask):
    """
    Post-traite le mask prédit

    Args:
        mask: Mask prédit par le modèle

    Returns:
        Mask au format liste Python
    """
    # Retirer la dimension batch
    mask = np.squeeze(mask, axis=0)

    # Convertir en classes (argmax si softmax)
    if mask.shape[-1] > 1:
        mask = np.argmax(mask, axis=-1)

    return mask.tolist()

@app.route('/')
def home():
    """Route de base pour vérifier que l'API fonctionne"""
    return jsonify({
        'status': 'online',
        'message': 'API de segmentation d\'images pour véhicules autonomes',
        'model_loaded': model is not None,
        'endpoints': {
            '/': 'GET - Information sur l\'API',
            '/health': 'GET - Vérification santé de l\'API',
            '/predict': 'POST - Prédiction de segmentation (multipart/form-data avec image)'
        }
    })

@app.route('/health')
def health():
    """Route de santé pour le monitoring"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route de prédiction de segmentation

    Attend une image en multipart/form-data
    Retourne le mask segmenté
    """
    # Vérifier que le modèle est chargé
    if model is None:
        return jsonify({
            'error': 'Modèle non chargé',
            'message': 'Le modèle de segmentation n\'a pas pu être chargé'
        }), 500

    # Vérifier qu'une image est présente
    if 'image' not in request.files:
        return jsonify({
            'error': 'Aucune image fournie',
            'message': 'Veuillez envoyer une image via le champ "image"'
        }), 400

    file = request.files['image']

    # Vérifier que le fichier n'est pas vide
    if file.filename == '':
        return jsonify({
            'error': 'Nom de fichier vide',
            'message': 'Le fichier envoyé n\'a pas de nom'
        }), 400

    try:
        # Lire l'image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Prétraiter l'image
        processed_image = preprocess_image(image)

        # Faire la prédiction
        prediction = model.predict(processed_image)

        # Post-traiter le mask
        mask = postprocess_mask(prediction)

        return jsonify({
            'status': 'success',
            'message': 'Segmentation effectuée avec succès',
            'mask': mask,
            'shape': {
                'height': len(mask),
                'width': len(mask[0]) if mask else 0
            }
        })

    except Exception as e:
        return jsonify({
            'error': 'Erreur lors de la prédiction',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Pour le développement local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
