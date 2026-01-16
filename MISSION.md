Comment allez-vous procéder ?
 
Cette mission suit un scénario de projet professionnel.
Vous pouvez suivre les étapes pour vous aider à réaliser vos livrables.

Avant de démarrer, nous vous conseillons de :

lire toute la mission et ses documents liés ;
prendre des notes sur ce que vous avez compris ;
consulter les étapes pour vous guider ; 
préparer une liste de questions pour votre première session de mentorat.

Prêt à mener la mission ? 
 
Future Vision Transport est une entreprise qui conçoit des systèmes embarqués de vision par ordinateur pour les véhicules autonomes. 

Vous êtes l’un des ingénieurs IA au sein de l’équipe R&D de cette entreprise. Votre équipe est composée d’ingénieurs aux profils variés. Chacun des membres de l’équipe est spécialisé sur une des parties du système embarqué de vision par ordinateur. 

Voici les différentes parties du système :

acquisition des images en temps réel
traitement des images
segmentation des images (c’est vous !)
système de décision
Vous travaillez sur la partie de segmentation des images (3) qui est alimentée par le bloc de traitement des images (2) et qui alimente le système de décision (4).

Votre rôle est de concevoir un premier modèle de segmentation d’images qui devra s’intégrer facilement dans la chaîne complète du système embarqué.

Lors d’une première phase de cadrage, vous avez récolté les avis de Franck et Laura, qui travaillent sur les parties avant et après votre intervention :

Franck, en charge du traitement des images (2) :

Le jeu de données que Franck utilise est disponible à ce lien, ou en téléchargement direct à ces liens : 1 ou 2, (images segmentées et annotées de caméras embarquées). On a uniquement besoin des 8 catégories principales (et non pas des 32 sous-catégories)
Laura, en charge du système de décision (4)

Souhaite une API simple à utiliser.
L’API prend en entrée une image et renvoie la segmentation de l’image de l’algo.
Exemple du jeu de données “Cityscapes”
Exemple d'image du jeu de données “Cityscapes”
Pour récapituler, vous avez dressé un plan d’action, avec les points suivants :

entraîner un modèle de segmentation des images sur les 8 catégories principales. Keras est le framework de travail commun à toute l’équipe. Attention aux contraintes de Franck !
concevoir une API de prédiction (Flask ou FastAPI) qui sera utilisée par Laura et la déployer sur le Cloud (Azure, Heroku, PythonAnywhere ou toute autre solution). Cette API prend en entrée une image et renvoie le mask prédit (segments prédits de l’image).
concevoir une application web (Flask, Streamlit) de présentation des résultats et la déployer sur le Cloud (Azure, Heroku, PythonAnywhere ou toute autre solution). Cette application sera l’interface pour tester l’API et afficher les images et masks.