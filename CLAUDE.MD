# CLAUDE.MD - Projet 8 OpenClassrooms

## Vue d'ensemble du projet

**Nom du projet**: Système de segmentation d'images pour véhicules autonomes
**Organisation**: Future Vision Transport
**Objectif**: Concevoir un modèle de segmentation d'images pour système embarqué de vision par ordinateur

## Contexte métier

Ce projet s'inscrit dans le développement d'un système embarqué de vision par ordinateur pour véhicules autonomes. Le système complet est composé de 4 modules :

1. **Acquisition des images** en temps réel
2. **Traitement des images** (Franck)
3. **Segmentation des images** ← VOTRE RÔLE
4. **Système de décision** (Laura)

### Contraintes des parties prenantes

**Franck (Traitement d'images)**:
- Dataset: Cityscapes (images de caméras embarquées)
- Lien: disponible dans les fichiers MISSION.md
- Utiliser uniquement les **8 catégories principales** (pas les 32 sous-catégories)

**Laura (Système de décision)**:
- Besoin d'une **API simple**
- Entrée: image
- Sortie: segmentation (mask prédit)

## Structure du projet

```
.
├── data/                      # Données du projet
│   ├── config.json           # Configuration des données
│   ├── gtFine/               # Ground truth (annotations)
│   ├── leftImg8bit/          # Images sources
│   ├── train_paths.csv       # Chemins images d'entraînement
│   ├── val_paths.csv         # Chemins images de validation
│   └── test_paths.csv        # Chemins images de test
├── notebooks/                 # Pipeline de développement
│   ├── 01_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_model_architecture.ipynb
│   └── 04_training.ipynb
├── src/                       # Code source
│   ├── models.py             # Architecture des modèles
│   └── utils.py              # Fonctions utilitaires
├── logs/                      # Logs d'entraînement
├── requirements.txt           # Dépendances Python
└── requirements-macbook.txt   # Dépendances macOS spécifiques
```

## Technologies utilisées

- **Framework ML**: Keras (standard de l'équipe R&D)
- **Dataset**: Cityscapes (8 catégories principales)
- **API**: Flask ou FastAPI (à déployer sur Cloud)
- **Application web**: Flask ou Streamlit (pour présentation résultats)
- **Déploiement**: Azure, Heroku, PythonAnywhere ou autre solution Cloud

## Livrables attendus

### 1. Pipeline complet dans notebooks ✓
- Scripts permettant l'exécution du pipeline complet
- Démonstration du caractère "industrialisable"
- Focus sur le générateur de données

### 2. API de prédiction (à déployer)
- **Framework**: Flask ou FastAPI
- **Entrée**: Image
- **Sortie**: Mask prédit (segments identifiés)
- **Déploiement**: Cloud (Azure/Heroku/PythonAnywhere)
- **Utilisateur final**: Laura (système de décision)

### 3. Application web de présentation (à déployer)
- **Framework**: Flask ou Streamlit
- **Fonctionnalités**:
  - Affichage de la liste des IDs d'images disponibles
  - Sélection d'une image par ID
  - Lancement de la prédiction via appel API
  - Affichage: image réelle + mask réel + mask prédit
- **Déploiement**: Cloud (Azure/Heroku/PythonAnywhere)

### 4. Note technique (~10 pages)
- Présentation des différentes approches
- Synthèse de l'état de l'art
- Présentation détaillée du modèle et architecture retenue
- Synthèse des résultats (incluant gains avec augmentation de données)
- Conclusion et pistes d'amélioration

### 5. Support de présentation (max 30 slides)
- Démarche méthodologique
- Présentation des résultats pour Laura

## État actuel du projet

### Notebooks développés
- ✅ `01_exploration.ipynb` - Exploration des données
- ✅ `02_data_preparation.ipynb` - Préparation et prétraitement
- ✅ `03_model_architecture.ipynb` - Conception architecture
- 🔄 `04_training.ipynb` - Entraînement (en cours)

### Code source
- ✅ `src/models.py` - Architectures de modèles
- ✅ `src/utils.py` - Fonctions utilitaires

### Données
- ✅ Dataset Cityscapes téléchargé et organisé
- ✅ Fichiers CSV de chemins générés (train/val/test)
- ✅ Configuration (config.json)

### À faire
- [ ] Finaliser l'entraînement du modèle
- [ ] Développer l'API de prédiction
- [ ] Créer l'application web de démonstration
- [ ] Déployer API et application sur le Cloud
- [ ] Rédiger la note technique
- [ ] Préparer le support de présentation

## Commandes utiles

### Environnement
```bash
# Activer l'environnement virtuel
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
# ou pour macOS
pip install -r requirements-macbook.txt
```

### Jupyter
```bash
# Lancer Jupyter
jupyter notebook
```

### Git
```bash
# Voir le statut
git status

# État actuel: fichiers modifiés
# - notebooks/01_exploration.ipynb
# - notebooks/02_data_preparation.ipynb
# - notebooks/03_model_architecture.ipynb
```

## Points d'attention

### Contraintes techniques
1. **Framework obligatoire**: Keras (standard équipe)
2. **8 catégories seulement**: Ne pas utiliser les 32 sous-catégories
3. **Compatibilité**: Attention aux contraintes de Franck (traitement d'images)
4. **Simplicité API**: Laura a besoin d'une interface simple

### Critères de qualité
- **Industrialisable**: Code production-ready
- **Générateur de données**: Optimisé pour l'entraînement
- **API robuste**: Gestion des erreurs et validation
- **Documentation**: Code bien documenté

### Augmentation de données
- Approches d'augmentation doivent être testées
- Gains à documenter dans la note technique

## Ressources

- **Dataset**: Cityscapes (voir MISSION.md pour liens)
- **Documentation projet**: MISSION.md, LIVRABLES.md
- **Catégories principales**: 8 classes de segmentation sémantique

## Notes de développement

### Choix d'architecture
- À documenter après expérimentation dans notebooks

### Résultats d'entraînement
- Métriques à ajouter après entraînement complet
- Comparaison des différentes approches

### Optimisations
- Générateur de données pour gestion mémoire
- Augmentation de données pour améliorer performance

---

**Dernière mise à jour**: 2026-01-29
**Statut**: En développement - Phase d'entraînement
