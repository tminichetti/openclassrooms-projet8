# Support de Présentation

## Système de Segmentation d'Images — Future Vision Transport

> **Instructions** : Ce fichier décrit chaque slide du support. À convertir en PowerPoint.
> Les sections marquées `[VISUEL]` indiquent les éléments graphiques à insérer.
> Les sections marquées `[À COMPLETER]` nécessitent les résultats d'entraînement.

---

### SLIDE 1 — Titre

**Titre** : Segmentation d'Images pour Véhicules Autonomes
**Sous-titre** : Présentation des résultats — Module Segmentation
**Visuel** : Logo Future Vision Transport + image Cityscapes en fond
**Bas de page** : Future Vision Transport — R&D — Février 2026

---

### SLIDE 2 — Agenda

**Titre** : Plan de la présentation

- Notre rôle dans le système
- Le dataset et la préparation des données
- Les architectures testées
- Les résultats
- Le déploiement (API + Application)
- Conclusion

---

### SLIDE 3 — Contexte : Le système embarqué

**Titre** : Notre rôle dans la chaîne

```
[1] Acquisition          [2] Traitement          [3] Segmentation          [4] Décision
des images      →       des images (Franck)  →   des images (Nous)    →    (Laura)
```

**Points clés** :
- On reçoit une image traitée de Franck
- On produit un mask de segmentation
- Laura l'utilise pour décider les actions du véhicule

`[VISUEL]` : Diagramme horizontal avec les 4 blocs, notre bloc en couleur accentuée

---

### SLIDE 4 — Objectifs

**Titre** : Ce qu'on a fait

- Entraîner un modèle de segmentation sur 8 catégories urbaines
- Comparer plusieurs architectures (simple → complexe)
- Mesurer l'apport de l'augmentation des données
- Livrer une API simple pour Laura
- Déployer tout en production

---

### SLIDE 5 — Le dataset : Cityscapes

**Titre** : Dataset Cityscapes

**Gauche** : Exemple d'image Cityscapes

**Droite** :
- Images de caméras embarquées, villes allemandes
- 5 000 images annotées au total
- 34 classes d'origine → regroupées en 8 pour nous
- Benchmark standard pour segmentation urbaine

`[VISUEL]` : Une image originale + son mask coloré côte à côte

---

### SLIDE 6 — Les 8 catégories

**Titre** : Notre classification

| Catégorie    | Couleur | Exemples                          |
|--------------|---------|-----------------------------------|
| Void         | Noir    | Arrière-plan                      |
| Flat         | Violet  | Route, trottoir                   |
| Construction | Gris    | Bâtiments, ponts                  |
| Object       | Bleu    | Poteaux, panneaux                 |
| Nature       | Vert    | Arbres, végétation                |
| Sky          | Blanc   | Ciel                              |
| Human        | Orange  | Piétons, cyclistes                |
| Vehicle      | Jaune   | Voitures, camions, bus            |

`[VISUEL]` : Exemple d'image avec mask coloré selon la palette ci-dessus

---

### SLIDE 7 — Séparation des données

**Titre** : Train / Validation / Test

```
[ ████████████████████████████████ Train : 2 975 images (60%) ]
[ ████████ Validation : 500 images (10%)                      ]
[ ████████████████ Test : 1 525 images (30%)                  ]
```

**Points clés** :
- Séparation officielle de Cityscapes (pas de mélange)
- Le test set n'est utilisé qu'à la fin pour l'évaluation finale
- Pas de fuite d'information entre les jeux

---

### SLIDE 8 — Pipeline général

**Titre** : Notre pipeline

```
Images brutes     Générateur      Entraînement      Évaluation     Déploiement
    +          →  de données   →   du modèle    →   sur test   →   API + Web
Masks (34 cls)    (8 classes)      + callbacks       + comparaison
```

`[VISUEL]` : Diagramme horizontal avec les 5 étapes

---

### SLIDE 9 — Générateur de données

**Titre** : Gestion des données à la volée

**Comment ça marche** :
- Classe Python héritant de `Sequence` (Keras)
- Chargement des images batch par batch en mémoire
- Conversion 34 classes → 8 catégories en temps réel (LUT)
- Traitement sur plusieurs cœurs de calcul automatiquement

**Pourquoi** :
- Le dataset est trop grand pour tout charger en mémoire
- Garantit une pipeline industrialisable et automatisée

`[VISUEL]` : Schéma : Fichier image → Chargement → Resize → LUT → Augmentation → Batch

---

### SLIDE 10 — Augmentation des données

**Titre** : Comment on augmente les données

| Technique       | Exemple                              | Pourquoi                           |
|-----------------|--------------------------------------|------------------------------------|
| Flip horizontal | L'image est mirée aléatoirement      | Plus de variété, même en miroir    |
| Luminance       | L'image est plus claire ou plus sombre | Conditions d'éclairage variées    |
| Contraste       | Les tones sont plus ou moins marqués | Robustesse aux conditions lumière  |

**Attention** : Le flip s'applique aussi au mask pour garder la cohérence !

`[VISUEL]` : 3 versions de la même image avec les 3 transformations

---

### SLIDE 11 — Callbacks

**Titre** : Comment on contrôle l'entraînement

**EarlyStopping** : On arrête si le modèle n'apprend plus (patience = 10 epochs)
**ModelCheckpoint** : On sauvegarde le meilleur modèle à chaque amélioration
**ReduceLROnPlateau** : On diminue le learning rate si la loss se plafonne

`[VISUEL]` : Courbe de loss avec les 3 callbacks annotés (meilleur point, arrêt, réduction LR)

---

### SLIDE 12 — Architecture U-Net

**Titre** : Notre modèle de base : U-Net

**Description** :
- Encodeur : 4 niveaux de downsampling
- Bottleneck : bloc central
- Décodeur : 4 niveaux d'upsampling
- Skip connections : préservent les détails fins
- Sortie : softmax sur 8 classes

**Chiffres** :
- 7,8M de paramètres (version light)
- Input : 256×512×3
- Output : 256×512×8

`[VISUEL]` : Diagramme classique U-Net avec les skip connections en couleur

---

### SLIDE 13 — Transfer Learning avec VGG16

**Titre** : On peut faire mieux : VGG16 + U-Net

**Qu'est-ce que c'est** :
- On prend VGG16, déjà entraîné sur ImageNet (1,2M d'images)
- On l'utilise comme encodeur à la place du notre
- Il connaît déjà les patterns visuels de base
- On entraîne juste le décodeur au début

**Avantages** :
- Convergence plus rapide
- Meilleure performance, même avec peu de données
- Poids initialisés avec des représentations riches

`[VISUEL]` : Diagramme U-Net avec l'encodeur VGG16 en couleur différente + "ImageNet" au-dessus

---

### SLIDE 14 — Loss functions testées

**Titre** : Comment on mesure l'erreur

| Loss           | Comment ça marche                                   | Adapté à |
|----------------|-----------------------------------------------------|----------|
| Cross-Entropy  | Classique, penalise chaque pixel                    | Baseline |
| Dice Loss      | Optimise le chevauchement prédiction/réalité        | Classes rares |
| Combined       | Cross-Entropy + Dice (notre choix)                  | Tout |
| Focal Loss     | Focus sur les pixels difficiles                     | Testée, moins bon |

`[VISUEL]` : Illustration visuelle de ce que fait chaque loss sur un exemple simple

---

### SLIDE 15 — Métriques d'évaluation

**Titre** : Comment on juge un bon modèle

**IoU (Jaccard)** : La métrique principale
```
IoU = Surface commune / Surface totale
      ██████               ██████████
      ██  ██   →  IoU =    ██████████  = 0.75 (par exemple)
      ██████               ██████████
   Prédit   Réel           Union
```

**Dice** : Très proche de l'IoU, tend à être un peu plus élevé

**mIoU** : On calcule l'IoU pour chaque classe, puis on fait la moyenne → la métrique clé

`[VISUEL]` : Illustration graphique de l'intersection et de l'union

---

### SLIDE 16 — Tableau des résultats

**Titre** : Comparaison des modèles

| Modèle          | Augmentation | Dice     | mIoU     | Accuracy |
|-----------------|--------------|----------|----------|----------|
| U-Net           | Non          | [À completer] | [À completer] | [À completer] |
| U-Net           | Oui          | [À completer] | [À completer] | [À completer] |
| U-Net + VGG16   | Oui          | [À completer] | [À completer] | [À completer] |

`[VISUEL]` : Graphique à barres colorées (vert = avec aug, rouge = sans aug)
Fichier disponible : `logs/comparison_metrics.png`

---

### SLIDE 17 — Impact de l'augmentation

**Titre** : L'augmentation ça fait quoi exactement ?

**Comparaison directe** (même modèle, avec et sans augmentation) :

| Métrique   | Sans augmentation | Avec augmentation | Gain    |
|------------|-------------------|-------------------|---------|
| Dice       | [À completer]     | [À completer]     | [À completer] |
| mIoU       | [À completer]     | [À completer]     | [À completer] |

**Ce qu'on observe** :
- Moins d'overfitting (gap train/val plus petit)
- Meilleure généralisation sur les données inconnues
- Gain en pourcentage à documenter

`[VISUEL]` : Fichier disponible : `logs/augmentation_impact.png`

---

### SLIDE 18 — Courbes d'apprentissage

**Titre** : Comment les modèles apprennent

`[VISUEL]` : Fichier disponible : `logs/learning_curves_comparison.png`

**Ce qu'on voit** :
- La loss diminue puis se plafonne
- Le Dice et mIoU augmentent
- L'EarlyStopping active au bon moment
- Pas d'overfitting majeur grâce à l'augmentation

---

### SLIDE 19 — Exemples de prédictions

**Titre** : Résultats qualitatifs

`[VISUEL]` : 3-4 exemples côte à côte :
- Image originale
- Mask ground truth (réel)
- Mask prédit par le modèle
- Overlay (image + mask coloré à 50%)

Fichier disponible : `logs/[experiment_name]/predictions_sample.png`

---

### SLIDE 20 — Le modèle retenu

**Titre** : Notre choix final

**Modèle** : [À completer — ex : U-Net + VGG16 avec augmentation]

**Pourquoi** :
- Meilleur Dice / mIoU parmi les modèles testés
- Bon équilibre performance / temps d'inférence
- Robuste grâce à l'augmentation des données

**Chiffres clés** :
- Dice : [À completer]
- mIoU : [À completer]
- Temps d'inférence : [À completer]

---

### SLIDE 21 — Architecture du système déployé

**Titre** : Comment tout est connecté

```
┌──────────────┐     HTTP POST     ┌─────────────┐
│  Streamlit   │  (image uploadée) │  API Flask  │
│  (Frontend)  │ ─────────────────>│  (Heroku)   │
│  Streamlit   │  (mask JSON)      │             │
│  Cloud       │ <─────────────────│  + Modèle   │
└──────────────┘                   └─────────────┘
                                        │
                                        │ Inférence
                                        ▼
                                   ┌─────────────┐
                                   │  Keras      │
                                   │  Model      │
                                   └─────────────┘
```

`[VISUEL]` : Diagramme propre avec les 3 composants et les flèches

---

### SLIDE 22 — L'API Flask

**Titre** : L'API que Laura peut utiliser

**Endpoints** :
- `GET /health` → Vérifie que l'API est en vie
- `POST /predict` → Envoie une image, reçoit le mask

**Comment l'utiliser** :
```python
import requests
response = requests.post(
    "https://api-url.herokuapp.com/predict",
    files={'image': open('image.jpg', 'rb')}
)
mask = response.json()['mask']
```

**Design** :
- Simple à utiliser (une seule requête)
- Indépendante de l'app web
- Robuste (gestion des erreurs)

---

### SLIDE 23 — L'application Streamlit

**Titre** : Interface de démonstration

**Fonctionnalités** :
- Upload une image
- Click "Lancer la segmentation"
- Voir : image originale + mask prédit + overlay
- Distribution des classes en pourcentage

`[VISUEL]` : Screenshot de l'application Streamlit en action

---

### SLIDE 24 — Demo

**Titre** : Démonstration en direct

`[VISUEL]` : Ici, montrer l'application en live ou un GIF/vidéo de la démonstration

> Si pas de demo live : montrer 3-4 screenshots de l'application avec différentes images

---

### SLIDE 25 — Ce qui reste à faire

**Titre** : Pistes d'amélioration

**Court terme** :
- Fine-tuner l'encodeur VGG16 (débloquer les couches)
- Tester ResNet50 / EfficientNet comme encodeur
- Augmentation plus diversifiée (rotation, zoom, élasticité)

**Moyen terme** :
- Augmenter la résolution d'entrée
- Optimiser le modèle pour l'embarqué (quantization, pruning)
- IoU par classe pour analyser les erreurs

**Long terme** :
- Explorer les Transformers (SegFormer)
- Tester en conditions adverses (nuit, pluie)
- Monitoring en production

---

### SLIDE 26 — Résumé

**Titre** : En un mot

- ✅ Pipeline complet : données → modèle → API → web
- ✅ Plusieurs architectures testées et comparées
- ✅ Impact de l'augmentation documenté
- ✅ API simple et déployée pour Laura
- ✅ Application de démonstration en production
- 🎯 Le modèle [À completer] retenu avec un Dice de [À completer]

---

### SLIDE 27 — Questions ?

**Titre** : Merci !

**Contact** : [Vos coordonnées]
**Dépôt** : [URL du GitHub]
**API** : [URL Heroku]
**App** : [URL Streamlit Cloud]

---

## Récapitulatif

| # Slides | Section                        | Nombre de slides |
|----------|--------------------------------|------------------|
| 1-2      | Titre + Agenda                 | 2                |
| 3-4      | Contexte + Objectifs           | 2                |
| 5-7      | Dataset                        | 3                |
| 8-11     | Méthodologie                   | 4                |
| 12-15    | Architectures + Loss + Métriques | 4              |
| 16-20    | Résultats                      | 5                |
| 21-24    | Déploiement + Demo             | 4                |
| 25-27    | Conclusion + Questions         | 3                |
| **Total**|                                | **27 slides**    |

> 3 slides de marge restantes si vous voulez en ajouter.
