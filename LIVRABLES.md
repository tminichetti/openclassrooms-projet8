Livrables 

Les scripts développés sur un notebook permettant l’exécution du pipeline complet :
    Ce livrable vous servira à présenter le caractère “industrialisable” de votre travail en particulier le générateur de données.
    
Une API (Flask ou FastAPI) déployée sur le Cloud (Azure, Heroku, PythonAnywhere ou toute autre solution), pour exposer votre modèle entraîné et qui recevra en entrée une image et retournera le mask prédit (les segments identifiés par votre modèle) :
    Ce livrable permettra à Laura d’utiliser facilement votre modèle.

Une application (Flask, Streamlit) de présentation des résultats qui consomme l’API de prédiction, déployée sur le Cloud (Azure, Heroku, PythonAnywhere ou toute autre solution). Cette application sera l’interface pour tester l’API et intégrera les fonctionnalités suivantes :  affichage de la liste des id des images disponibles, lancement de la prédiction du mask pour l’id sélectionné par appel à l’API, et affichage de l’image réelle, du mask réel et du mask prédit :
    Ce livrable permettra d’illustrer votre travail auprès de vos collègues

Une note technique de 10 pages environ contenant une présentation des différentes approches et une synthèse de l’état de l’art, la présentation plus détaillée du modèle et de l’architecture retenue, une synthèse des résultats obtenus (incluant les gains obtenus avec les approches d’augmentation des données) et une conclusion avec des pistes d’amélioration envisageables  :
    Ce livrable vous servira à présenter votre démarche technique à vos collègues.

Un support de présentation (type Power Point) de votre démarche méthodologique (30 slides maximum) :
    Ce livrable vous permettra de présenter vos résultats à Laura.