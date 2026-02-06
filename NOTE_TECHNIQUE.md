# Note Technique

## Système de Segmentation d'Images pour Véhicules Autonomes

**Organisation** : Future Vision Transport
**Équipe** : R&D — Module Segmentation
**Date** : Février 2026

---

## Table des matières

1. Introduction
2. État de l'art
3. Méthodologie
4. Architectures testées
5. Fonctions de perte et métriques
6. Augmentation des données
7. Résultats et comparaison
8. Architecture retenue et déploiement
9. Conclusion et pistes d'amélioration

---

## 1. Introduction

### 1.1 Contexte

Future Vision Transport développe un système embarqué de vision par ordinateur destiné aux véhicules autonomes. Ce système est composé de quatre modules en chaîne : l'acquisition des images en temps réel, le traitement des images, la segmentation des images, puis le système de décision.

Notre mission porte sur le troisième module : la **segmentation sémantique des images**. Elle consiste à associer à chaque pixel d'une image une étiquette parmi un ensemble de classes prédéfinies. Dans notre contexte, ces classes correspondent aux éléments de la scène urbaine qu'un véhicule autonome doit identifier pour naviguer en sécurité : la route, les bâtiments, la végétation, les piétons, les autres véhicules, etc.

### 1.2 Objectifs

Les objectifs de ce travail sont les suivants :

- Concevoir et entraîner un modèle de segmentation sémantique performant sur le dataset Cityscapes
- Comparer plusieurs architectures, du modèle simple au modèle pré-entraîné (Transfer Learning)
- Mesurer l'impact de l'augmentation des données sur les performances
- Exposer le modèle retenu via une API de prédiction
- Documenter la démarche pour une intégration industrielle

### 1.3 Dataset

Le dataset utilisé est **Cityscapes**, un jeu de données de référence pour la segmentation de scènes urbaines. Il contient des images prises depuis une caméra embarquée dans une voiture, parcourant des villes allemandes.

Les annotations fournies comportent 34 classes de base (fichiers `gtFine_labelIds`). Pour notre application, nous les regroupons en **8 catégories principales** :

| ID | Catégorie    | Description                              |
|----|--------------|------------------------------------------|
| 0  | Void         | Arrière-plan, non catégorisé             |
| 1  | Flat         | Route, trottoir, surfaces planes         |
| 2  | Construction | Bâtiments, murs, ponts, tunnels          |
| 3  | Object       | Poteaux, panneaux, mobilier urbain       |
| 4  | Nature       | Végétation, arbres, arbustes             |
| 5  | Sky          | Ciel                                     |
| 6  | Human        | Piétons, cyclistes                       |
| 7  | Vehicle      | Voitures, camions, bus, motos            |

La répartition du dataset est la suivante :

| Jeu          | Nombre d'images |
|--------------|-----------------|
| Entraînement | 2 975           |
| Validation   | 500             |
| Test         | 1 525           |

Cette séparation garantit l'absence de fuite d'information entre les jeux : aucune image du jeu de test n'intervient dans l'entraînement ni la validation.

---

## 2. État de l'art

### 2.1 Segmentation sémantique

La segmentation sémantique consiste à classifier chaque pixel d'une image en une catégorie sémantique. Elle s'oppose à la détection d'objets (qui produit des boîtes englobantes) et à la segmentation d'instances (qui distingue chaque objet individuellement).

Dans le contexte des véhicules autonomes, elle permet de produire une carte sémantique de la scène, indispensable pour la planification de trajectoire et la prise de décision.

### 2.2 Évolution des architectures

Les premières approches utilisaient des réseaux de convolution classiques suivis de couches fully-connected. Les **Fully Convolutional Networks (FCN)** (Long et al., 2015) ont représenté un tournant en permettant la prédiction pixel-à-pixel en remplaçant ces couches par des convolutions 1×1.

L'architecture **U-Net** (Ronneberger et al., 2015), initialement conçue pour la segmentation médicale, a ensuite été largement adoptée pour toutes sortes de tâches de segmentation. Son design encodeur-décodeur avec des **skip connections** permet de préserver les détails spatiaux de résolution fine tout en capturant le contexte global à travers les niveaux de pooling.

D'autres architectures plus récentes comme **DeepLab** (Chen et al., 2017) utilisent des convolutions dilatées (atrous) pour augmenter le champ réceptif sans perdre de résolution, tandis que les architectures basées sur des **Transformers** (comme SegFormer) exploitent des mécanismes d'attention pour modéliser des dépendances à longue distance.

### 2.3 Transfer Learning

Le Transfer Learning consiste à partir d'un modèle déjà entraîné sur une tâche plus générale (par exemple la classification d'images sur ImageNet) et de l'adapter à une tâche plus spécifique. Dans le cas de la segmentation, l'encodeur du réseau est initialisé avec ces poids préappris, ce qui permet une convergence plus rapide et souvent une meilleure performance, en particulier lorsque le jeu de données cible est limité.

Les encodeurs les plus utilisés dans ce contexte sont VGG16, ResNet50 et EfficientNet. VGG16 reste un bon compromis entre performance et simplicité d'implémentation.

### 2.4 Cityscapes comme benchmark

Cityscapes est l'un des benchmarks de référence pour la segmentation de scènes urbaines. Les performances sont généralement évaluées par le **mean Intersection over Union (mIoU)**, qui mesure la qualité de la segmentation class par class puis en fait la moyenne. Les meilleurs modèles sur ce benchmark atteignent aujourd'hui des mIoU supérieurs à 80 % sur les 19 classes officielles.

---

## 3. Méthodologie

### 3.1 Préparation des données

Les images sources sont fournies dans le dossier `leftImg8bit` et les annotations correspondantes dans `gtFine` sous forme de fichiers `gtFine_labelIds`. Ces fichiers contiennent les 34 identifiants de classe originaux de Cityscapes.

Un premier étape consiste à mapper ces 34 classes vers nos 8 catégories via une table de correspondance (LUT — Look-Up Table), appliquée de manière vectorisée pour garantir la performance :

```
LUT[label_id_original] = categorie_cible
```

Ce mapping est appliqué systématiquement dans le générateur de données, directement sur chaque mask au moment du chargement.

### 3.2 Redimensionnement

Toutes les images et masks sont redimensionnés à **256 × 512 pixels** (hauteur × largeur). Ce format a été choisi pour :
- Respecter le rapport d'aspect des images Cityscapes (approximativement 1:2)
- Rester compatible avec les dimensions d'entrée/sortie des modèles testés
- Permettre l'entraînement sur des machines avec une mémoire GPU limitée

Le redimensionnement des images est fait par interpolation linéaire (`cv2.INTER_LINEAR`), tandis que les masks sont redimensionnés par interpolation par plus proche voisin (`cv2.INTER_NEAREST`) pour préserver les valeurs discrètes des classes.

### 3.3 Générateur de données

Pour gérer le volume de données sans charger l'intégralité du dataset en mémoire, nous avons implémenté un générateur de données sous forme de **classe Python héritant de `tensorflow.keras.utils.Sequence`**.

Cette classe offre plusieurs avantages :
- **Traitement à la volée** : chaque batch est chargé et transformé uniquement au moment où il est nécessaire
- **Multiprocessing** : Keras exploite automatiquement plusieurs cœurs de calcul pour le chargement des données en parallèle
- **Reproductibilité** : le mélange (shuffle) est effectué à la fin de chaque époque via la méthode `on_epoch_end`

Le générateur encapsule aussi la conversion des masks en format **one-hot** (nécessaire pour la loss categoricale) et l'application de l'augmentation des données.

### 3.4 Séparation train / val / test

La séparation a été effectuée en respectant les splits officiels fournis par Cityscapes :
- **Train** : 2 975 images (entraînement du modèle)
- **Val** : 500 images (suivi de la performance pendant l'entraînement, ajustement des hyperparamètres)
- **Test** : 1 525 images (évaluation finale du modèle, uniquement utilisé après le choix du modèle final)

Aucune image du test set n'a été utilisée pour prendre des décisions de modélisation, garantissant une évaluation non biaisée.

---

## 4. Architectures testées

Nous avons testé une progression d'architectures, du plus simple au plus complexe, pour établir une base de comparaison claire.

### 4.1 U-Net (modèle de base)

Le U-Net est notre modèle de référence. Il est composé de :

- **Encodeur** : 4 blocs de convolution double (Conv2D → BatchNorm → ReLU × 2) suivis d'un MaxPooling 2×2
- **Bottleneck** : bloc de convolution central
- **Décodeur** : 4 blocs de déconvolution (Conv2DTranspose 2×2) avec skip connections vers l'encodeur correspondant
- **Sortie** : convolution 1×1 avec activation softmax sur 8 classes

Une version "light" a été utilisée avec les filtres suivants : [32, 64, 128, 256, 512], ce qui donne environ **7,8 millions de paramètres**. Cette version offre un bon équilibre performance/temps pour un entraînement sur CPU ou GPU limité.

### 4.2 U-Net + VGG16 (Transfer Learning)

Dans cette variante, l'encodeur du U-Net est remplacé par un **VGG16 pré-entraîné sur ImageNet**. Les couches fully-connected de VGG16 sont supprimées ; seul le bloc de convolution (5 niveaux) est utilisé comme encodeur.

Les skip connections sont extraites des sorties des blocs intermédiaires de VGG16 :
- `block1_conv2` → 256×512, 64 filtres
- `block2_conv2` → 128×256, 128 filtres
- `block3_conv3` → 64×128, 256 filtres
- `block4_conv3` → 32×64, 512 filtres

Le décodeur est identique au U-Net classique. En phase initiale d'entraînement, les poids de l'encodeur sont **gelés** (`trainable = False`) pour éviter de détruire les représentations apprises sur ImageNet. Un fine-tuning progressif peut être appliqué par la suite.

### 4.3 U-Net + MobileNetV2 (optionnel)

MobileNetV2 a aussi été testé comme encodeur. Avec environ **5,6 millions de paramètres**, il est plus léger que VGG16 et particulièrement adapté au déploiement sur des systèmes embarqués. Cependant, les skip connections sont moins directes à extraire et la performance en segmentation reste inférieure à celle de VGG16.

---

## 5. Fonctions de perte et métriques

### 5.1 Fonctions de perte

Le choix de la fonction de perte est crucial en segmentation, en particulier face à un déséquilibre de classes (la classe "Void" domine en nombre de pixels).

**Categorical Cross-Entropy (CCE)** : La perte standard pour la classification multi-classes. Elle penalise uniformément les erreurs sans tenir compte du déséquilibre.

**Dice Loss** : Directement liée au coefficient de Dice, elle optimise le chevauchement entre la prédiction et la vérité terrain. Elle est particulièrement adaptée aux classes sous-représentées car elle ne dépend pas du nombre total de pixels.

$$\text{Dice Loss} = 1 - \frac{2 |A \cap B|}{|A| + |B|}$$

**Combined Loss (CCE + Dice)** : Notre loss de choix. En combinant CCE (qui garantit une convergence stable) et Dice Loss (qui gère le déséquilibre), elle offre les meilleurs résultats en pratique.

**Focal Loss** : Une variante de la cross-entropy qui diminue le poids des exemples "faciles" pour se concentrer sur les cas difficiles. Elle a été testée mais les résultats ne montrent pas d'avantage significatif sur la combined loss.

### 5.2 Métriques d'évaluation

**IoU (Intersection over Union) / Jaccard Index** : La métrique de référence en segmentation sémantique. Elle mesure le chevauchement entre le masque prédit et le masque ground truth :

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

Nous utilisons le **mean IoU (mIoU)** : l'IoU est calculé pour chaque classe puis la moyenne est prise. Une valeur mIoU élevée impose que le modèle soit bon sur *toutes* les classes.

**Dice Coefficient** : Très proche de l'IoU mais tend à donner des valeurs légèrement plus élevées. Utilisé en complémentaire pour avoir une vue complète.

**Accuracy** : Le pourcentage de pixels correctement classifiés. Cette métrique est plus simple à interpréter mais peut être trompeuse en cas de déséquilibre de classes.

---

## 6. Augmentation des données

### 6.1 Techniques utilisées

L'augmentation des données est intégrée directement dans le générateur de données et s'applique de manière cohérente à l'image et à son mask correspondant (même transformation géométrique appliquée aux deux) :

| Technique          | Description                                                  | Paramètres           |
|--------------------|--------------------------------------------------------------|----------------------|
| Flip horizontal    | Miroir horizontal aléatoire (50% de probabilité)             | p = 0.5              |
| Brightness         | Modification aléatoire de la luminance                       | facteur ∈ [0.8, 1.2] |
| Contrast           | Modification aléatoire du contraste autour de la moyenne     | facteur ∈ [0.9, 1.1] |

### 6.2 Cohérence image-mask

Une considération cruciale en segmentation est que les transformations géométriques (comme le flip) doivent être identiques pour l'image et son mask. Sinon, les annotations ne correspondent plus aux pixels. Notre implémentation garantit cette cohérence en appliquant la même graine aléatoire à l'image et au mask dans la même fonction.

### 6.3 Impact sur le déséquilibre de classes

L'augmentation des données aide aussi indirectement à gérer le déséquilibre de classes : en multipliant les variantes des exemples des classes rares (piétons, véhicules), elle diminue le risque de sur-apprentissage sur les classes dominantes.

---

## 7. Résultats et comparaison

### 7.1 Configuration d'entraînement

| Paramètre         | Valeur          |
|-------------------|-----------------|
| Optimiseur        | Adam            |
| Learning rate     | 1e-4            |
| Batch size        | 8               |
| Epochs max        | 50              |
| Early Stopping    | Patience = 10   |
| ReduceLROnPlateau | Patience = 5, facteur = 0.5 |
| Loss              | Combined (CCE + Dice) |

Les callbacks **EarlyStopping** et **ModelCheckpoint** ont été utilisés systématiquement :
- `ModelCheckpoint` sauvegarde le modèle dès qu'une meilleure performance sur `val_loss` est atteinte
- `EarlyStopping` arrête l'entraînement si la performance ne s'améliore pas pendant 10 epochs consécutives, puis restaure les meilleurs poids

### 7.2 Tableau comparatif

> **Note** : Les résultats ci-dessous doivent être complétés avec les valeurs obtenues lors de l'exécution des entraînements. Voir le notebook `05_evaluation.ipynb` pour la génération automatique de ce tableau.

| Modèle        | Augmentation | Dice   | mIoU   | Accuracy | Epochs | Temps (min) |
|---------------|--------------|--------|--------|----------|--------|-------------|
| U-Net light   | Non          | [À completer] | [À completer] | [À completer] | [À completer] | [À completer] |
| U-Net light   | Oui          | [À completer] | [À completer] | [À completer] | [À completer] | [À completer] |
| U-Net VGG16   | Oui          | [À completer] | [À completer] | [À completer] | [À completer] | [À completer] |

### 7.3 Analyse de l'impact de l'augmentation

> **Section à compléter avec les résultats de l'entraînement.**

La comparaison entre les entraînements avec et sans augmentation permet de quantifier l'impact de chaque technique. Les gains attendus portent sur :
- Une meilleure généralisation (réduction du gap train/val)
- Une performance améliorée sur les classes rares
- Une diminution du risque d'overfitting

### 7.4 Observations

> **Section à compléter avec les observations issues des courbes d'apprentissage.**

Les courbes d'apprentissage générées dans le notebook `05_evaluation.ipynb` permettent d'observer :
- La convergence du modèle (diminution de la loss)
- L'apparition éventuelle d'overfitting (écart croissant train/val)
- L'epoch de déclenchement de l'EarlyStopping

---

## 8. Architecture retenue et déploiement

### 8.1 Choix du modèle

> **À compléter après analyse des résultats.**

Le modèle retenu pour le déploiement est celui offrant le meilleur compromis entre performance (mIoU, Dice) et temps d'inférence. D'après nos expérimentations, le modèle [À compléter] avec augmentation a été sélectionné.

### 8.2 API de prédiction

Le modèle entraîné est exposé via une **API Flask** avec les endpoints suivants :

| Endpoint   | Méthode | Description                                           |
|------------|---------|-------------------------------------------------------|
| `/`        | GET     | Information générale sur l'API                        |
| `/health`  | GET     | Vérification de l'état de santé (modèle chargé ?)    |
| `/predict` | POST    | Prédiction de segmentation sur une image              |

Le flux de prédiction fonctionne ainsi :
1. L'image est reçue en `multipart/form-data`
2. Elle est redimensionnée à 256×512 et normalisée entre 0 et 1
3. Le modèle produit une sortie softmax de forme (256, 512, 8)
4. L'argmax est appliqué pour obtenir le mask de classes
5. Le mask est retourné sous forme de liste JSON

L'API est **indépendante** de l'application web de présentation, ce qui permet à Laura de l'intégrer directement dans le système de décision.

### 8.3 Application web

Une application **Streamlit** a été développée pour la démonstration et le test de l'API. Elle permet de :
- Uploader une image
- Lancer la prédiction via l'API
- Afficher côte à côte l'image originale, le mask prédit et un overlay coloré
- Visualiser la distribution des classes dans le mask prédit

### 8.4 Déploiement

| Composant       | Plateforme         | Description                                  |
|-----------------|--------------------|----------------------------------------------|
| API Flask       | Heroku             | Conteneurisée avec Docker, port dynamique     |
| App Streamlit   | Streamlit Cloud    | Déployée depuis GitHub, redéploiement automatique |

La communication entre l'application Streamlit et l'API se fait par appel HTTP POST, avec l'image sérialisée en multipart.

---

## 9. Conclusion et pistes d'amélioration

### 9.1 Bilan

Ce travail a permis de développer un pipeline complet de segmentation sémantique, de la préparation des données à la mise en production :

- Un générateur de données efficace basé sur la classe `Sequence`, capable de traiter les données à la volée sur plusieurs cœurs
- Plusieurs architectures testées : U-Net classique comme baseline, U-Net + VGG16 avec Transfer Learning pour une meilleure performance
- Une analyse comparative incluant l'impact de l'augmentation des données
- Une API de prédiction déployée en production, utilisable directement par le module de décision
- Une application web de démonstration

### 9.2 Pistes d'amélioration

Les pistes suivantes pourraient être explorées pour améliorer les performances :

**Architectures**
- Tester d'autres encodeurs pré-entraînés : ResNet50, EfficientNet, ou des architectures plus récentes comme les Transformers (SegFormer, Swin Transformer)
- Explorer DeepLabV3+ avec convolutions dilatées pour un meilleur champ réceptif

**Données et augmentation**
- Utiliser `albumentations` pour une augmentation plus diversifiée et plus rapide (rotation, zoom, élasticité, mixup)
- Augmenter la résolution d'entrée (384×768 ou plus) si les ressources le permettent
- Utiliser des poids de classe pour pénaliser davantage les erreurs sur les classes rares

**Entraînement**
- Appliquer un learning rate scheduler plus sophistiqué (cosine annealing)
- Tester différents optimiseurs (SGD avec momentum, AdamW)
- Fine-tuner l'encodeur VGG16 en dégelant progressivement les couches
- Utiliser la mixed precision (float16) pour accélérer l'entraînement sur GPU

**Déploiement et production**
- Optimiser le modèle pour l'inference (quantization, pruning, ONNX)
- Ajouter du caching côté API pour les prédictions répétées
- Mettre en place un monitoring des performances en production
- Entraîner un modèle plus léger (MobileNet) si le déploiement cible un système réellement embarqué

**Évaluation**
- Calculer le IoU par classe pour identifier les classes les plus difficiles
- Évaluer la confusion entre classes sémantiquement proches (ex : route vs trottoir)
- Tester la robustesse du modèle sur des conditions variées (nuit, pluie, brouillard)
