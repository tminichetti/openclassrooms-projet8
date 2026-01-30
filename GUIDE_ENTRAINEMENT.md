# Guide d'Entraînement des Modèles

Ce guide explique comment entraîner les modèles de segmentation et comparer les résultats.

## 📋 Prérequis

1. **Données préparées** :
   - `data/train_paths.csv`
   - `data/val_paths.csv`
   - `data/config.json`

2. **Dépendances installées** :
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU recommandé** (mais fonctionne sur CPU)

---

## 🚀 Entraînement des Modèles

### Option 1 : Script Python (Recommandé)

Le script `train.py` permet d'entraîner les modèles facilement depuis la ligne de commande.

#### Entraîner U-Net simple SANS augmentation
```bash
python train.py --model unet --no-augmentation --epochs 30
```

#### Entraîner U-Net simple AVEC augmentation
```bash
python train.py --model unet --augmentation --epochs 30
```

#### Entraîner VGG16 SANS augmentation
```bash
python train.py --model vgg16 --no-augmentation --epochs 30
```

#### Entraîner VGG16 AVEC augmentation
```bash
python train.py --model vgg16 --augmentation --epochs 30
```

#### Options disponibles
```bash
python train.py --help

Options:
  --model {unet,vgg16}     Type de modèle
  --augmentation           Activer data augmentation
  --no-augmentation        Désactiver data augmentation
  --epochs N               Nombre d'epochs (défaut: 30)
  --batch-size N           Taille des batches (défaut: 8)
  --learning-rate LR       Learning rate (défaut: 0.0001)
  --patience N             Patience early stopping (défaut: 10)
```

### Option 2 : Notebook Jupyter

Si vous préférez utiliser les notebooks :

```bash
jupyter notebook notebooks/04_training.ipynb
```

Puis exécuter toutes les cellules.

---

## 📊 Comparaison des Résultats

Après avoir entraîné **au moins 2 modèles**, comparer les résultats.

### Option 1 : Notebook Jupyter (Recommandé ⭐)

```bash
jupyter notebook notebooks/05_evaluation.ipynb
```

Puis exécuter toutes les cellules. Le notebook va :
- ✅ Afficher un tableau comparatif interactif
- ✅ Analyser l'impact de l'augmentation
- ✅ Identifier le meilleur modèle
- ✅ Générer 5 graphiques professionnels
- ✅ Exporter les tableaux (LaTeX pour note technique)
- ✅ Analyser les courbes d'apprentissage

**Avantage** : Visualisations interactives + documentation intégrée, parfait pour la note technique !

### Option 2 : Script Python (vérification rapide)

```bash
python compare_models.py
```

**Avantage** : Plus rapide pour un check en ligne de commande

### Fichiers générés

```
logs/
├── all_results.csv           # Tous les résultats d'entraînement
├── comparison.png            # Graphiques comparatifs
├── comparison_table.md       # Tableau Markdown (pour GitHub)
└── comparison_table.tex      # Tableau LaTeX (pour note technique)
```

---

## 🎯 Plan d'Entraînement Recommandé

Pour satisfaire **Milestone 3**, voici le plan minimal :

### Phase 1 : Tests rapides (2-3 epochs) ⚡
Vérifier que tout fonctionne :
```bash
python train.py --model unet --no-augmentation --epochs 3
python train.py --model unet --augmentation --epochs 3
```

### Phase 2 : Entraînements complets 🚀

**Configuration minimale (CPU/GPU faible)** :
```bash
# U-Net sans augmentation (baseline)
python train.py --model unet --no-augmentation --epochs 20 --batch-size 4

# U-Net avec augmentation
python train.py --model unet --augmentation --epochs 20 --batch-size 4

# VGG16 avec augmentation (meilleur modèle attendu)
python train.py --model vgg16 --augmentation --epochs 20 --batch-size 4
```

**Configuration optimale (GPU performant)** :
```bash
# U-Net sans augmentation
python train.py --model unet --no-augmentation --epochs 50 --batch-size 8

# U-Net avec augmentation
python train.py --model unet --augmentation --epochs 50 --batch-size 8

# VGG16 sans augmentation
python train.py --model vgg16 --no-augmentation --epochs 40 --batch-size 8

# VGG16 avec augmentation
python train.py --model vgg16 --augmentation --epochs 40 --batch-size 8
```

### Phase 3 : Analyse des résultats 📈
```bash
python compare_models.py
```

---

## 📂 Organisation des Fichiers

Après l'entraînement, voici l'organisation :

```
.
├── train.py                          # Script d'entraînement
├── compare_models.py                 # Script de comparaison
├── models/                           # Modèles sauvegardés
│   ├── unet_no-aug_YYYYMMDD_HHMMSS_best.keras
│   ├── unet_aug_YYYYMMDD_HHMMSS_best.keras
│   ├── vgg16_no-aug_YYYYMMDD_HHMMSS_best.keras
│   └── vgg16_aug_YYYYMMDD_HHMMSS_best.keras
└── logs/                             # Logs d'entraînement
    ├── all_results.csv               # Résultats consolidés
    ├── comparison.png                # Graphiques
    ├── comparison_table.md           # Tableau Markdown
    ├── unet_no-aug_YYYYMMDD_HHMMSS/
    │   ├── config.json               # Config expérience
    │   ├── training_log.csv          # Log par epoch
    │   ├── history.csv               # Historique complet
    │   ├── training_curves.png       # Courbes d'apprentissage
    │   └── results.json              # Résultats finaux
    ├── unet_aug_YYYYMMDD_HHMMSS/
    ├── vgg16_no-aug_YYYYMMDD_HHMMSS/
    └── vgg16_aug_YYYYMMDD_HHMMSS/
```

---

## ⏱️ Temps d'Entraînement Estimés

### Sur CPU (macOS M1/M2 ou équivalent)
- U-Net (30 epochs) : **~2-3 heures**
- VGG16 (30 epochs) : **~3-4 heures**

### Sur GPU (NVIDIA RTX 3060 ou équivalent)
- U-Net (30 epochs) : **~20-30 minutes**
- VGG16 (30 epochs) : **~30-45 minutes**

### Sur Google Colab (GPU gratuit)
- U-Net (30 epochs) : **~30-40 minutes**
- VGG16 (30 epochs) : **~45-60 minutes**

---

## 🎓 Utiliser Google Colab (GPU gratuit)

Si vous n'avez pas de GPU local :

### 1. Créer un notebook Colab

Aller sur : https://colab.research.google.com/

### 2. Activer GPU

Runtime → Change runtime type → GPU

### 3. Installer les dépendances

```python
!git clone https://github.com/votre-username/openclassrooms-projet8.git
%cd openclassrooms-projet8
!pip install -r requirements.txt
```

### 4. Uploader les données

Soit via Google Drive, soit via :
```python
from google.colab import files
# Upload data/config.json, data/train_paths.csv, data/val_paths.csv
```

### 5. Lancer l'entraînement

```python
!python train.py --model unet --augmentation --epochs 30
```

### 6. Télécharger les résultats

```python
from google.colab import files
files.download('models/unet_aug_YYYYMMDD_HHMMSS_best.keras')
files.download('logs/all_results.csv')
```

---

## 🐛 Dépannage

### Erreur : Out of Memory (OOM)

**Solution 1** : Réduire batch size
```bash
python train.py --model unet --augmentation --batch-size 4
```

**Solution 2** : Réduire taille des images
Éditer `data/config.json` :
```json
{
  "img_height": 128,
  "img_width": 256
}
```

### Erreur : Module not found

Installer les dépendances :
```bash
pip install tensorflow pandas numpy opencv-python matplotlib
```

### L'entraînement est trop lent sur CPU

Options :
1. Utiliser Google Colab (GPU gratuit)
2. Réduire le nombre d'epochs
3. Réduire le nombre de données d'entraînement (pour tests)

### Callbacks ne s'arrêtent pas

Vérifier la patience :
```bash
python train.py --model unet --augmentation --patience 5
```

---

## 📈 Métriques à Surveiller

### Pendant l'entraînement

1. **Loss** (doit diminuer) :
   - Train loss < Val loss = normal
   - Écart trop grand = overfitting

2. **Dice Coefficient** (doit augmenter) :
   - > 0.70 = bon modèle
   - > 0.80 = très bon modèle

3. **Mean IoU** (doit augmenter) :
   - > 0.60 = bon modèle
   - > 0.70 = très bon modèle

4. **Accuracy** (doit augmenter) :
   - > 0.85 = bon modèle
   - > 0.90 = très bon modèle

### Après l'entraînement

Comparer :
- **Impact augmentation** : gain attendu de 2-5% sur Dice
- **Transfer learning** : VGG16 devrait être meilleur que U-Net simple
- **Temps d'entraînement** : Trade-off performance/temps

---

## ✅ Checklist Milestone 3

Avant de passer à la suite, vérifier :

- [ ] Au moins **2 modèles** entraînés (U-Net + VGG16 recommandé)
- [ ] Comparaison **avec/sans augmentation** (minimum 1 modèle)
- [ ] **Tableau comparatif** généré (`logs/comparison_table.md`)
- [ ] **Graphiques** générés (`logs/comparison.png`)
- [ ] **Meilleur modèle** identifié
- [ ] **EarlyStopping** et **ModelCheckpoint** utilisés (automatique dans le script)
- [ ] **Gains augmentation** documentés
- [ ] **Temps d'entraînement** documentés

---

## 🚀 Prochaines Étapes (Milestone 6)

Après l'entraînement :

1. **Copier le meilleur modèle** dans l'API :
   ```bash
   cp models/MEILLEUR_MODELE.keras api/model/segmentation_model.h5
   ```

2. **Tester l'API** localement :
   ```bash
   cd api
   python app.py
   python test_api.py
   ```

3. **Déployer sur Heroku** :
   ```bash
   cd api
   heroku create nom-api
   git push heroku main
   ```

4. **Déployer Streamlit** :
   - Push sur GitHub
   - Déployer sur Streamlit Cloud
   - Configurer API_URL

5. **Rédiger note technique** avec :
   - Tableau comparatif
   - Graphiques
   - Analyse des résultats
   - Recommandations

---

## 📞 Support

Si vous rencontrez des problèmes :

1. Vérifier les logs dans `logs/`
2. Vérifier les fichiers de config
3. Consulter le fichier ANALYSE_NOTEBOOKS.md
4. Consulter le fichier MILESTONES.md

Bon courage ! 🎓🚀
