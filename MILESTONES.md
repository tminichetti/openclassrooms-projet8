Étapes clés

 
#Milestone 1 : Conception des modèles de segmentation d’images
Livrable : 
 Notebook partie « conception des modèles »
Problèmes et erreurs courants :
Problème de sélection des bons types d’images « mask » (target des modèles) dans les datas fournies
Problèmes de compatibilité des librairies / codes réutilisés (Tensorflow / Keras)
Recommandations :
Les images target « mask » à prendre dans le dataset sont celles nommées « gtFine_labelIds », qui contiennent 34 valeurs différentes, correspondant aux 34 classes de base (d’où le gris foncé de l’image à l’affichage). Il convient dans la « class » de type « sequence » (traitement à la volée des images) de transformer les 34 classes en 8 catégories telles que définies
L’étudiant pourra prendre par exemple un modèle simple non pré-entraîné comme « Unet mini » et un modèle pré-entraîné comme « VGG16 Unet » (encoder = VGG16 pré-entraîné), CF « Ressources »
L’étudiant pourra tester différents types de loss, comme le Dice_loss, le total_loss, ou balanced_cross_entropy
Les mesures principales dans ce contexte sont l’IoU (Jaccard) et le Dice_coeff
Souvent les exemples utilisent des anciennes versions de Keras et Tensorflow. Il convient de tout passer en « tensorflow.keras.xxx » lors des imports afin d’assurer une compatibilité dans les traitements
 
#Milestone 2 : Préparation du générateur de données (Class de type Sequence)
Livrable : 
Notebook partie « préparation du flux de données » (« class » de type « Sequence »), y compris la partie « augmentation des données »
Problèmes et erreurs courants :
Problème de redimensionnement des images et masks
Recommandations :
La dimension des images réelles (« X ») doit être égale à la dimension d’entrée du modèle
La dimension des images « masks » (target « y ») doit être égale à la dimension de sortie du modèle (« mask » prédit)
Intégrer le data augmentation dans ce traitement des images à la volée, via des fonctions internes ou l’utilisation de librairie telle « albumentations » ou “imgaug”
 
#Milestone 3 : Entraînement des modèles et comparaison
Livrable : 
Notebook partie « simulations » avec tableau récapitulatif
Problèmes et erreurs courants :
Temps de traitement et limitation de ressources en local
Recommandations :
Exécution d’un notebook sur le studio du WorkSpace Azure ML créé
Ou bien exécution d’un script Python à distance
Utiliser un « EarlyStopping » afin d’arrêter le traitement au bout de N epochs sans amélioration
Utiliser en complémentarité un « ModelCheckPoint » pour sauvegarder en fin d’epoch le modèle s’il est meilleur que les précédents
 
#Milestone 4 : Création de l’application Flask
Livrable : 
Code Python de l’API Flask ou FastAPI de prédiction du mask
 
#Milestone 5 : Création de l’application Web Flask
Livrable : 
Code Python de l’application Flask qui gère les interaction, appelle l’API de prédiction et affiche le mask prédit ainsi que l’image réelle et le mask réel
Recommandations :
L’objectif étant d’utiliser des solutions gratuites pour les déploiements, un set très limité de données images peut être stocké directement dans l’application Web Flask au lieu normalement de les stocker sur un compte de stockage Cloud
Les échanges entre l’application Flask et l’API peuvent se faire par sérialisation / désérialisation des images (CF “Ressources”)
Autre variante : 
Création d’une API qui expose plusieurs fonctionnalités :
Affichage des id des images disponibles
Récupération des images réelle (image et mask déjà annoté)
Prédiction du mask à partir de l’image réelle, par appel le moteur de prédiction
Création d’une application Streamlit locale, de gestion de l’interaction avec l’utilisateur et d’affichage des images et masks, par appel aux fonctionnalités de l’API
 
#Milestone 6 : Mise en production de l’application Flask et de l’API
Livrable : 
API déployée sur le Cloud
Application Flask déployée sur le Cloud et accessible via une url
Problèmes et erreurs courants :
Gestion des requirements  (requirements.txt)
Recommandations :
L’étudiant a le choix de la solution Cloud gratuite pour le déploiement de l’API :
Azure webapp (App Service)
Choisir l’option de l’App Service Plan avec « Sku and size » égal idéalement à « Free F1 »  gratuit (code + packages < 1 Go) 
Création sur le portail Azure d’une webapp, avec mise à jour du code manuellement, ou automatiquement via une source telle que Github
Ou création en ligne de commande d’une webapp via « az webapp up », à partir du répertoire de l’application Flask
Heroku ou PythonAnywhere 
Autres solutions au choix
L’étudiant a la possibilité d’optimiser son déploiement en créant un conteneur Docker : https://openclassrooms.com/fr/courses/2035766-optimisez-votre-deploiement-en-creant-des-conteneurs-avec-docker  
 

 

#Critères d’évaluation de la mission
Barre.png

 

#Définir la stratégie d’élaboration d’un modèle d'apprentissage profond, concevoir ou ré-utiliser des modèles pré-entraînés (transfer learning) et entraîner des modèles afin de réaliser une analyse prédictive.
CE1 Le candidat a défini sa stratégie d’élaboration d’un modèle pour répondre à un besoin métier (par exemple : choix de conception d’un modèle ou ré-utilisation de modèles pré-entraînés).

CE2 Le candidat a identifié la ou les cibles. 

CE3 Le candidat a réalisé la séparation du jeu de données en jeu d’entraînement, jeu de validation et jeu de test. 

CE4 Le candidat s'est assuré qu'il n’y a pas de fuite d’information entre les jeux de données (entraînement, validation et test). 

CE5 Le candidat a testé plusieurs modèles d’apprentissage profond en partant du plus simple vers les plus complexes. Dans le cadre de ce projet, par exemple : 

un modèle simple, tel que le unet_mini
un modèle intégrant un encodeur pré-entrainé, tel qu’un VGG16 Unet
CE6 Le candidat a mis en oeuvre des modèles à partir de modèles pré-entraînés (technique de Transfer Learning)

 

#Évaluer la performance des modèles d’apprentissage profond selon différents critères (scores, temps d'entraînement, etc.) afin de choisir le modèle le plus performant pour la problématique métier.
CE1 Le candidat a choisi une métrique adaptée à la problématique métier, et sert à évaluer la performance des modèles. Dans le cadre du projet : 

par exemple : IoU,  Dice_coef
CE2 Le candidat a explicité le choix de la métrique d’évaluation 

CE3 Le candidat a évalué la performance d’un modèle de référence et sert de comparaison pour évaluer la performance des modèles plus complexes 

CE4 Le candidat a calculé, hormis la métrique choisie, au moins un autre indicateur pour comparer les modèles (par exemple : le temps nécessaire pour l’entraînement du modèle) 

CE5 Le candidat a optimisé au moins un des hyperparamètres du modèle choisi (par exemple : le choix de la fonction Loss, le Batch Size, le nombre d'Epochs) 

CE6 Le candidat a présenté une synthèse comparative des différents modèles, par exemple sous forme de tableau. 

CE7 Le candidat a déployé le modèle de machine learning sous forme d'API (via Flask par exemple) et cette API renvoie bien une prédiction correspondant à une demande. Dans le cadre du projet : 

Le modèle prend en entrée une image sérialisée et retourne l’image sérialisée des segments identifiés par le modèle (mask)
CE8 Le candidat a réalisé l'API indépendamment de l'application qui utilise le résultat de la prédiction. 

CE9 Le candidat a défini, préparé et mis en œuvre un pipeline de déploiement continu, afin de déployer l'API et l’application web sur un serveur d'une plateforme Cloud. Il a notamment créé un dossier pour chaque application (application web et API) contenant tous les scripts, dans un logiciel de version de code (ex : Git) et l'a partagé (ex : Github)

 

#Utiliser des techniques d’augmentation des données afin d'améliorer la performance des modèles.
CE1 Le candidat a utilisé plusieurs techniques d’augmentation des données (ex. pour des images : rotation, changement d’échelle, ajout de bruit…). 

CE2 Le candidat a présenté une synthèse comparative des améliorations de performance grâce aux différentes techniques d'augmentation de données utilisées (maîtrise de l’overfitting, meilleure performance).

 

#Manipuler un jeu de données volumineux
CE1 Le candidat a développé et testé un générateur de données permettant le traitement des images sur plusieurs cœurs de calcul.

CE2  Le candidat a développé le générateur de données sous forme de classe Python.

CE3  Le candidat a entièrement automatisé le script du générateur de données.