# Application Streamlit - Démonstration de Segmentation

Application web Streamlit pour visualiser et tester le système de segmentation d'images pour véhicules autonomes.

## Structure

```
streamlit/
├── app.py                  # Application Streamlit
├── requirements.txt        # Dépendances Python
├── .streamlit/            # Configuration Streamlit
│   └── config.toml        # Thème et paramètres
├── .env.example           # Exemple de configuration locale
└── README.md              # Ce fichier
```

## Fonctionnalités

### 🎯 Principales fonctionnalités

1. **Upload d'images**
   - Support des formats PNG, JPG, JPEG
   - Affichage de l'image originale avec informations

2. **Prédiction de segmentation**
   - Appel à l'API de prédiction sur Heroku
   - Affichage du mask coloré par classes
   - Overlay image + mask

3. **Visualisation des résultats**
   - Mask de segmentation colorisé (8 classes Cityscapes)
   - Distribution des classes (pourcentages)
   - Comparaison côte à côte

4. **Configuration dynamique**
   - URL de l'API modifiable
   - Vérification de l'état de l'API
   - Légende des classes avec couleurs

## Installation et utilisation

### 1. Test local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py

# Avec une URL d'API spécifique
API_URL=https://votre-api.herokuapp.com streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

### 2. Déploiement sur Streamlit Cloud

#### 2.1 Prérequis

- Compte GitHub : https://github.com
- Compte Streamlit Cloud : https://streamlit.io/cloud (connexion avec GitHub)
- Code poussé sur un dépôt GitHub

#### 2.2 Préparation du dépôt

```bash
# Depuis la racine du projet
cd streamlit

# S'assurer que le code est sur GitHub
git add .
git commit -m "Add Streamlit app"
git push origin main
```

#### 2.3 Déploiement

1. **Aller sur Streamlit Cloud**
   - https://share.streamlit.io/

2. **Se connecter avec GitHub**

3. **Créer une nouvelle app**
   - Cliquer sur "New app"
   - Sélectionner votre dépôt GitHub
   - Branch : `main` (ou votre branche)
   - Main file path : `streamlit/app.py`

4. **Configurer les secrets (IMPORTANT)**
   - Cliquer sur "Advanced settings"
   - Dans "Secrets", ajouter :
   ```toml
   API_URL = "https://votre-api-segmentation.herokuapp.com"
   ```

5. **Déployer**
   - Cliquer sur "Deploy!"
   - Attendre que l'application se lance (1-2 minutes)

6. **Votre app est en ligne !**
   - URL automatique : `https://votre-username-nom-repo-streamlit-app-hash.streamlit.app/`
   - Vous pouvez personnaliser l'URL dans les settings

#### 2.4 Mise à jour de l'application

Streamlit Cloud redéploie automatiquement à chaque push sur GitHub :

```bash
# Modifier app.py
git add streamlit/app.py
git commit -m "Update Streamlit app"
git push origin main
# → L'app se redéploie automatiquement !
```

## Configuration de l'URL de l'API

### En local

**Option 1 : Variable d'environnement**
```bash
export API_URL=https://votre-api.herokuapp.com
streamlit run app.py
```

**Option 2 : Fichier .env** (créer à partir de .env.example)
```bash
cp .env.example .env
# Éditer .env avec votre URL
streamlit run app.py
```

**Option 3 : Directement dans l'interface**
- L'URL peut être modifiée dans la sidebar de l'application

### Sur Streamlit Cloud

**Via les Secrets (RECOMMANDÉ)**

1. Dans le dashboard Streamlit Cloud
2. Cliquer sur votre app → "⋮" → "Settings"
3. Section "Secrets"
4. Ajouter :
```toml
API_URL = "https://votre-api.herokuapp.com"
```
5. Sauvegarder (l'app redémarre automatiquement)

## Utilisation de l'application

### 1. Vérifier l'API

1. Dans la sidebar, vérifier l'URL de l'API
2. Cliquer sur "Vérifier l'API"
3. S'assurer que l'API est accessible et le modèle chargé

### 2. Tester une segmentation

1. Cliquer sur "Browse files" pour uploader une image
2. Attendre l'affichage de l'image originale
3. Cliquer sur "🚀 Lancer la segmentation"
4. Visualiser les résultats :
   - Mask de segmentation colorisé
   - Distribution des classes
   - Comparaison et overlay

### 3. Interpréter les résultats

**Classes Cityscapes (8 catégories principales)**
- **Void/Background** : Arrière-plan / non catégorisé
- **Flat** : Routes, trottoirs, surfaces planes
- **Construction** : Bâtiments, murs, ponts
- **Object** : Poteaux, panneaux, mobilier urbain
- **Nature** : Végétation, arbres
- **Sky** : Ciel
- **Human** : Piétons, cyclistes
- **Vehicle** : Voitures, camions, bus

## Architecture

```
┌─────────────────┐
│   Utilisateur   │
└────────┬────────┘
         │ Upload image
         v
┌─────────────────┐
│   Streamlit     │
│  (Cloud Gratuit)│
└────────┬────────┘
         │ POST /predict
         v
┌─────────────────┐
│   API Flask     │
│   (Heroku)      │
└────────┬────────┘
         │ Inférence
         v
┌─────────────────┐
│   Modèle Keras  │
│   (segmentation)│
└─────────────────┘
```

## Personnalisation

### Modifier les couleurs des classes

Dans `app.py`, modifier le dictionnaire `CITYSCAPES_COLORS` :

```python
CITYSCAPES_COLORS = {
    0: [R, G, B],  # Votre couleur RGB
    1: [R, G, B],
    # ...
}
```

### Modifier le thème Streamlit

Éditer `.streamlit/config.toml` :

```toml
[theme]
primaryColor = "#FF6B6B"  # Couleur principale
backgroundColor = "#FFFFFF"  # Fond
secondaryBackgroundColor = "#F0F2F6"  # Fond secondaire
textColor = "#262730"  # Texte
```

## Dépannage

### L'API n'est pas accessible

1. Vérifier que l'URL de l'API est correcte (avec https://)
2. Vérifier que l'API est déployée et en ligne sur Heroku
3. Tester l'API directement : `curl https://votre-api.herokuapp.com/health`
4. Vérifier les secrets dans Streamlit Cloud

### Timeout lors de la prédiction

- Le timeout est fixé à 30 secondes
- Si le modèle est trop lent, optimiser l'inférence côté API
- Sur Heroku free tier, les dynos peuvent être en veille (première requête lente)

### Erreur de connexion CORS

- Normalement géré par l'API Flask
- Si problème, vérifier que l'API accepte les requêtes depuis Streamlit Cloud

### L'app ne se met pas à jour

- Forcer le redéploiement : Settings → "Reboot app"
- Vérifier que le code est bien poussé sur GitHub
- Vérifier les logs dans Streamlit Cloud

## Avantages de Streamlit Cloud

✅ **Gratuit** pour les projets publics
✅ **Déploiement automatique** depuis GitHub
✅ **Pas de configuration serveur** (vs Heroku)
✅ **URL propre** et personnalisable
✅ **Redémarrage automatique** à chaque push
✅ **Gestion des secrets** intégrée
✅ **Logs** accessibles dans le dashboard

## Limites de Streamlit Cloud

⚠️ **Ressources limitées** (1 CPU, 1GB RAM)
⚠️ **Peut s'endormir** après inactivité (comme Heroku free)
⚠️ **Dépôt public requis** (pour version gratuite)

## Workflow complet

1. **Entraîner le modèle** localement
2. **Déployer l'API** sur Heroku avec le modèle
3. **Tester l'API** avec curl ou test_api.py
4. **Pousser le code Streamlit** sur GitHub
5. **Déployer sur Streamlit Cloud** en pointant vers le dépôt
6. **Configurer l'URL de l'API** dans les secrets
7. **Tester l'application** en ligne ! 🎉

## Support

Pour les problèmes ou questions :
1. Vérifier les logs dans le dashboard Streamlit Cloud
2. Tester l'API séparément avec curl
3. Vérifier les secrets : Settings → Secrets
4. Documentation Streamlit : https://docs.streamlit.io/

## Ressources utiles

- **Streamlit Cloud** : https://streamlit.io/cloud
- **Documentation** : https://docs.streamlit.io/
- **Galerie d'exemples** : https://streamlit.io/gallery
- **Forum communautaire** : https://discuss.streamlit.io/
