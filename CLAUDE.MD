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

## Milestones du projet

### ✅ Milestone 1 : Conception des modèles
**Livrable** : Notebook partie conception des modèles
- ✅ Utiliser images `gtFine_labelIds` (34 classes → 8 catégories)
- ✅ Tester modèle simple : Unet mini
- ✅ Tester modèle pré-entraîné : VGG16 Unet (Transfer Learning)
- ✅ Tester différentes loss : Dice_loss, total_loss, balanced_cross_entropy
- ✅ Métriques : IoU (Jaccard) et Dice_coeff
- ✅ Utiliser `tensorflow.keras.xxx` pour compatibilité

### ✅ Milestone 2 : Générateur de données
**Livrable** : Notebook avec classe Sequence + augmentation
- ✅ Classe Python de type `Sequence` pour générateur
- ✅ Redimensionnement correct (X → entrée modèle, y → sortie modèle)
- ✅ Augmentation de données intégrée (albumentations ou imgaug)
- ✅ Traitement sur plusieurs cœurs de calcul
- ✅ Script entièrement automatisé

### 🔄 Milestone 3 : Entraînement et comparaison
**Livrable** : Notebook simulations + tableau récapitulatif
- 🔄 Entraînement des modèles (local ou Azure ML Studio)
- 🔄 Utiliser EarlyStopping (arrêt après N epochs sans amélioration)
- 🔄 Utiliser ModelCheckpoint (sauvegarde meilleur modèle)
- 🔄 Tableau comparatif des modèles (performances + temps)
- 🔄 Synthèse gains avec augmentation de données

### ✅ Milestone 4 : Création de l'API
**Livrable** : Code Python API Flask/FastAPI
- ✅ API Flask créée dans `/api`
- ✅ Endpoints : `/`, `/health`, `/predict`
- ✅ Entrée : image | Sortie : mask prédit
- ✅ Dockerfile pour déploiement Heroku

### ✅ Milestone 5 : Application Web
**Livrable** : Application Web d'interaction
- ✅ Application Streamlit créée dans `/streamlit`
- ✅ Upload d'images
- ✅ Appel API de prédiction
- ✅ Affichage : image réelle + mask prédit + overlay
- ✅ Distribution des classes
- ✅ Prête pour déploiement Streamlit Cloud

### ⏳ Milestone 6 : Mise en production
**Livrable** : Déploiements Cloud
- [ ] API déployée sur Heroku (avec modèle)
- [ ] Application Streamlit déployée sur Streamlit Cloud
- [ ] URLs accessibles et fonctionnelles
- [ ] Tests de bout en bout

## État actuel du projet

### Notebooks développés
- ✅ `01_exploration.ipynb` - Exploration des données
- ✅ `02_data_preparation.ipynb` - Préparation et prétraitement
- ✅ `03_model_architecture.ipynb` - Conception architecture
- 🔄 `04_training.ipynb` - Entraînement (structure prête, à exécuter)
- ✅ `05_evaluation.ipynb` - Évaluation et comparaison des modèles

### Code source
- ✅ `src/models.py` - Architectures de modèles
- ✅ `src/utils.py` - Fonctions utilitaires
- ✅ `api/app.py` - API Flask de prédiction
- ✅ `streamlit/app.py` - Application Web Streamlit

### Infrastructure déploiement
- ✅ `api/` - Dossier API avec Dockerfile Heroku
- ✅ `streamlit/` - Dossier Streamlit Cloud ready
- ✅ Documentation complète (READMEs)

### Données
- ✅ Dataset Cityscapes téléchargé et organisé
- ✅ Fichiers CSV de chemins générés (train/val/test)
- ✅ Configuration (config.json)

### À faire (priorité)
- [x] Analyser la conformité des notebooks avec les milestones
- [x] Créer script d'entraînement Python standalone (`train.py`)
- [x] Créer script de comparaison des modèles (`compare_models.py`)
- [x] Documenter le processus d'entraînement (`GUIDE_ENTRAINEMENT.md`)
- [ ] **Exécuter les entraînements** (Milestone 3 - CRITIQUE) :
  - [ ] U-Net sans augmentation (baseline)
  - [ ] U-Net avec augmentation
  - [ ] VGG16 avec augmentation (recommandé)
- [ ] Générer le tableau comparatif avec `compare_models.py`
- [ ] Copier le meilleur modèle → `api/model/segmentation_model.h5`
- [ ] Déployer l'API sur Heroku (Milestone 6)
- [ ] Déployer l'application sur Streamlit Cloud (Milestone 6)
- [ ] Tester le système complet bout-en-bout
- [ ] Rédiger la note technique (~10 pages)
- [ ] Préparer le support de présentation (max 30 slides)

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

## Critères d'évaluation (points clés)

### 1. Stratégie d'élaboration du modèle
- ✅ Définir stratégie (modèle simple vs pré-entraîné)
- ✅ Identifier les cibles (8 catégories Cityscapes)
- ✅ Séparation train/val/test (pas de fuite d'information)
- ✅ Tester plusieurs modèles (simple → complexe)
  - Exemple : Unet mini → VGG16 Unet
- ✅ Transfer Learning implémenté

### 2. Évaluation de la performance
- ✅ **Métrique principale** : IoU ou Dice_coef (adapté à la segmentation)
- ✅ Expliciter le choix de la métrique
- ✅ Modèle de référence pour comparaison
- 🔄 **Indicateurs complémentaires** : temps d'entraînement, mémoire
- 🔄 Optimisation hyperparamètres (Loss, Batch Size, Epochs)
- 🔄 **Tableau comparatif** des modèles
- ⏳ **API déployée** : entrée image → sortie mask
- ✅ API indépendante de l'application web
- ✅ Pipeline déploiement continu (Git/GitHub + Cloud)

### 3. Augmentation de données
- ✅ Plusieurs techniques testées (rotation, échelle, bruit, etc.)
- 🔄 **Synthèse comparative** des gains de performance
- 🔄 Impact sur l'overfitting documenté

### 4. Manipulation de données volumineuses
- ✅ **Générateur de données** développé (classe Sequence)
- ✅ Traitement sur plusieurs cœurs de calcul
- ✅ Script entièrement automatisé

## Points d'attention

### Contraintes techniques
1. **Framework obligatoire** : Keras (standard équipe)
2. **Images target** : Utiliser `gtFine_labelIds` (34 classes → 8 catégories)
3. **8 catégories seulement** : Ne pas utiliser les 32 sous-catégories
4. **Compatibilité** : Tout en `tensorflow.keras.xxx` pour éviter les conflits
5. **Redimensionnement** :
   - Dimension X = dimension d'entrée du modèle
   - Dimension y (mask) = dimension de sortie du modèle
6. **Simplicité API** : Laura a besoin d'une interface simple

### Critères de qualité
- **Industrialisable** : Code production-ready
- **Générateur de données** : Optimisé pour l'entraînement (multicore)
- **API robuste** : Gestion des erreurs et validation
- **Documentation** : Code bien documenté
- **Callbacks** : EarlyStopping + ModelCheckpoint obligatoires

### Augmentation de données
- Approches d'augmentation doivent être testées
- Gains à documenter dans la note technique
- Utiliser `albumentations` ou `imgaug` recommandé

## Workflow de déploiement

### Phase 1 : Entraînement (en cours)
```bash
# 1. Finaliser l'entraînement dans le notebook 04_training.ipynb
# 2. S'assurer d'avoir EarlyStopping et ModelCheckpoint
# 3. Sauvegarder le meilleur modèle
model.save('best_model.h5')
```

### Phase 2 : Préparation API
```bash
# 1. Copier le modèle dans le dossier API
cp best_model.h5 api/model/segmentation_model.h5

# 2. Tester l'API localement
cd api
pip install -r requirements.txt
python app.py

# 3. Tester avec le script de test
python test_api.py http://localhost:5000 chemin/vers/image.jpg
```

### Phase 3 : Déploiement API sur Heroku
```bash
cd api
heroku login
heroku create nom-api-segmentation
heroku stack:set container
git init
git add .
git commit -m "Deploy API"
git push heroku main

# Noter l'URL : https://nom-api-segmentation.herokuapp.com
```

### Phase 4 : Déploiement Streamlit Cloud
```bash
# 1. Pousser le code sur GitHub
git add streamlit/
git commit -m "Add Streamlit app"
git push origin main

# 2. Aller sur https://share.streamlit.io/
# 3. Créer une nouvelle app pointant vers streamlit/app.py
# 4. Configurer les secrets avec l'URL de l'API Heroku
# 5. Déployer !
```

### Phase 5 : Tests bout-en-bout
```bash
# 1. Tester l'API sur Heroku
curl https://nom-api-segmentation.herokuapp.com/health

# 2. Uploader une image dans Streamlit
# 3. Vérifier que la prédiction fonctionne
# 4. Documenter les URLs dans la note technique
```

## Ressources

### Dataset et documentation
- **Dataset** : Cityscapes (voir MISSION.md pour liens)
- **Documentation projet** : MISSION.md, LIVRABLES.md, MILESTONES.md
- **Catégories principales** : 8 classes de segmentation sémantique

### Modèles recommandés
- **Modèle simple** : Unet mini (baseline)
- **Modèle pré-entraîné** : VGG16 Unet (Transfer Learning)
- **Autres encodeurs** : ResNet, EfficientNet (optionnel)

### Loss functions à tester
- `Dice_loss` : Adapté à la segmentation
- `total_loss` : Combinaison de plusieurs loss
- `balanced_cross_entropy` : Pour classes déséquilibrées

### Métriques d'évaluation
- **IoU (Jaccard)** : Intersection over Union (métrique principale)
- **Dice coefficient** : Similaire à IoU, souvent utilisé en segmentation
- **Temps d'entraînement** : Pour comparer l'efficacité

### Librairies d'augmentation
- `albumentations` : Recommandé, performant
- `imgaug` : Alternative valide

## Notes de développement

### Choix d'architecture
- À documenter après expérimentation dans notebooks

### Résultats d'entraînement
- Métriques à ajouter après entraînement complet
- Comparaison des différentes approches

### Optimisations
- Générateur de données pour gestion mémoire
- Augmentation de données pour améliorer performance

## Récapitulatif de l'avancement

```
Progression globale du projet : ████████████░░░░░░░░ 60%

Milestones:
✅ M1 - Conception modèles          [████████████████████] 100%
✅ M2 - Générateur de données       [████████████████████] 100%
🔄 M3 - Entraînement                [████████████░░░░░░░░]  60%
✅ M4 - API Flask                   [████████████████████] 100%
✅ M5 - Application Web             [████████████████████] 100%
⏳ M6 - Déploiement Cloud           [░░░░░░░░░░░░░░░░░░░░]   0%

Livrables:
✅ Notebooks (01-03)                [████████████████████] 100%
🔄 Notebook 04 (training)           [████████████░░░░░░░░]  60%
✅ API Flask + Dockerfile           [████████████████████] 100%
✅ App Streamlit                    [████████████████████] 100%
⏳ Note technique                   [░░░░░░░░░░░░░░░░░░░░]   0%
⏳ Support présentation             [░░░░░░░░░░░░░░░░░░░░]   0%
```

### Prochaines étapes prioritaires

1. **Finaliser l'entraînement** (Milestone 3)
   - Compléter notebook `04_training.ipynb`
   - Générer tableau comparatif des modèles
   - Documenter gains avec augmentation de données

2. **Sauvegarder le modèle**
   - Exporter le meilleur modèle → `api/model/segmentation_model.h5`

3. **Déployer sur le Cloud** (Milestone 6)
   - API sur Heroku
   - Streamlit sur Streamlit Cloud
   - Tests bout-en-bout

4. **Rédiger les livrables finaux**
   - Note technique (~10 pages)
   - Support de présentation (max 30 slides)

---

**Dernière mise à jour** : 2026-01-30
**Statut** : En développement - Phase d'entraînement (Milestone 3)
**Infrastructure** : Prête pour déploiement (API + Streamlit)
