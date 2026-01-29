# API de Segmentation d'Images

API Flask pour la prédiction de segmentation d'images dans le cadre du projet de véhicules autonomes.

## Structure

```
api/
├── Dockerfile              # Configuration Docker
├── app.py                  # Application Flask
├── requirements.txt        # Dépendances Python
├── .dockerignore          # Fichiers à ignorer par Docker
├── model/                 # Dossier pour le modèle entraîné
│   └── segmentation_model.h5  # Modèle Keras (à ajouter)
└── README.md              # Ce fichier
```

## Installation et utilisation

### 1. Préparer le modèle

Après avoir entraîné votre modèle, copiez-le dans le dossier `model/` :

```bash
mkdir -p model
cp /chemin/vers/votre/modele.h5 model/segmentation_model.h5
```

### 2. Test local (sans Docker)

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
python app.py
```

L'API sera accessible sur `http://localhost:5000`

### 3. Test avec Docker (local)

```bash
# Construire l'image
docker build -t segmentation-api .

# Lancer le conteneur
docker run -p 5000:5000 -e PORT=5000 segmentation-api
```

### 4. Déploiement sur Heroku

#### 4.1 Prérequis

- Compte Heroku : https://signup.heroku.com/
- Heroku CLI installé : https://devcenter.heroku.com/articles/heroku-cli

```bash
# Vérifier l'installation
heroku --version
```

#### 4.2 Connexion et création de l'app

```bash
# Se connecter à Heroku
heroku login

# Créer une nouvelle application
heroku create nom-de-votre-app

# Ou utiliser un nom auto-généré
heroku create
```

#### 4.3 Configuration pour Heroku

```bash
# Se placer dans le dossier api
cd api

# Configurer le stack container
heroku stack:set container -a nom-de-votre-app

# Initialiser git si nécessaire
git init
git add .
git commit -m "Initial commit"

# Ajouter le remote Heroku
heroku git:remote -a nom-de-votre-app
```

#### 4.4 Déploiement

```bash
# Déployer sur Heroku
git push heroku main

# Ou si vous êtes sur une autre branche
git push heroku votre-branche:main
```

#### 4.5 Vérifier le déploiement

```bash
# Ouvrir l'application dans le navigateur
heroku open

# Voir les logs
heroku logs --tail

# Vérifier le statut
heroku ps
```

## Endpoints de l'API

### GET /

Informations générales sur l'API

**Réponse:**
```json
{
  "status": "online",
  "message": "API de segmentation d'images pour véhicules autonomes",
  "model_loaded": true,
  "endpoints": {...}
}
```

### GET /health

Vérification de l'état de santé de l'API

**Réponse:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict

Prédiction de segmentation sur une image

**Requête:**
- Method: POST
- Content-Type: multipart/form-data
- Body: fichier image avec le champ "image"

**Exemple avec curl:**
```bash
curl -X POST -F "image=@chemin/vers/image.jpg" http://localhost:5000/predict
```

**Exemple avec Python:**
```python
import requests

url = "http://localhost:5000/predict"
files = {'image': open('image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Réponse:**
```json
{
  "status": "success",
  "message": "Segmentation effectuée avec succès",
  "mask": [[0, 1, 2, ...], ...],
  "shape": {
    "height": 256,
    "width": 512
  }
}
```

## Notes importantes

### Taille du modèle

⚠️ **Attention** : Les fichiers sur Heroku sont limités à 500 MB (slug size). Si votre modèle est très volumineux :

1. Considérez la compression du modèle
2. Utilisez un stockage externe (S3, Google Cloud Storage)
3. Chargez le modèle depuis un URL au démarrage

### Variables d'environnement

Le port est automatiquement défini par Heroku via la variable `PORT`.

### Logs

Pour déboguer sur Heroku :
```bash
heroku logs --tail -a nom-de-votre-app
```

## Améliorations possibles

- [ ] Ajouter l'authentification (API key)
- [ ] Implémenter un cache pour les prédictions
- [ ] Ajouter des métriques (temps de réponse, etc.)
- [ ] Supporter différents formats de sortie (JSON, image PNG)
- [ ] Ajouter la validation des dimensions d'image
- [ ] Implémenter le batching pour plusieurs images

## Dépannage

### Le modèle ne se charge pas

Vérifiez que :
- Le fichier `model/segmentation_model.h5` existe
- Le modèle est compatible avec la version de Keras/TensorFlow

### Erreur de mémoire sur Heroku

- Utilisez un dyno avec plus de RAM
- Optimisez la taille du modèle
- Réduisez les dimensions d'entrée

### Timeout sur Heroku

- Les requêtes doivent se terminer en moins de 30 secondes
- Optimisez le temps d'inférence du modèle
