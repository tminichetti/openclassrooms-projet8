# Dockerfile pour l'API de segmentation d'images
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY app.py .
COPY model/ ./model/

# Exposer le port (Heroku définira le PORT via variable d'environnement)
EXPOSE $PORT

# Commande pour lancer l'application
CMD gunicorn --bind 0.0.0.0:$PORT app:app
