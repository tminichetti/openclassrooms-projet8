"""
Script pour extraire les résultats d'un notebook 04 déjà exécuté
et les ajouter au fichier logs/all_results.csv

Usage:
    python extract_results_from_notebook.py
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def extract_from_logs():
    """Extrait les résultats depuis les dossiers logs/"""

    LOGS_DIR = Path('logs')
    MODELS_DIR = Path('models')

    if not LOGS_DIR.exists():
        print("❌ Dossier logs/ introuvable")
        return

    all_results = []

    # Parcourir tous les dossiers d'expériences
    for exp_dir in LOGS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue

        # Chercher les fichiers nécessaires
        history_file = exp_dir / 'history.csv'
        config_file = exp_dir / 'config.json'

        if not history_file.exists() or not config_file.exists():
            print(f"⚠️  Ignorer {exp_dir.name} (fichiers manquants)")
            continue

        print(f"📂 Extraction depuis: {exp_dir.name}")

        # Charger l'historique
        history = pd.read_csv(history_file)

        # Charger la config
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Trouver le meilleur epoch
        best_epoch = history['val_loss'].idxmin()

        # Déterminer le modèle depuis le nom de l'expérience
        exp_name = exp_dir.name.lower()
        if 'vgg' in exp_name:
            model_name = 'vgg16'
        elif 'mobile' in exp_name:
            model_name = 'mobilenet'
        else:
            model_name = 'unet'

        # Déterminer si augmentation depuis le nom
        augmentation = 'aug' in exp_name and 'no-aug' not in exp_name

        # Chercher le fichier modèle
        model_files = list(MODELS_DIR.glob(f'*{exp_dir.name}*.keras'))
        if not model_files:
            model_files = list(MODELS_DIR.glob('*best.keras'))
        model_path = str(model_files[0]) if model_files else 'unknown'

        # Créer l'entrée
        result_entry = {
            'experiment': exp_dir.name,
            'model': model_name,
            'augmentation': augmentation,
            'epochs_trained': len(history),
            'best_epoch': int(best_epoch + 1),
            'training_time_minutes': 0,  # Non disponible depuis historique
            'val_loss': float(history.loc[best_epoch, 'val_loss']),
            'val_accuracy': float(history.loc[best_epoch, 'val_accuracy']),
            'val_dice': float(history.loc[best_epoch, 'val_dice_coefficient']),
            'val_miou': float(history.loc[best_epoch, 'val_mean_iou']),
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }

        all_results.append(result_entry)

        print(f"  ✅ {model_name} ({'avec' if augmentation else 'sans'} aug) - "
              f"Dice: {result_entry['val_dice']:.4f}")

    if not all_results:
        print("\n❌ Aucun résultat trouvé")
        print("\nAssurez-vous d'avoir exécuté le notebook 04 et qu'il y a des dossiers dans logs/")
        return

    # Créer le DataFrame
    df_results = pd.DataFrame(all_results)

    # Sauvegarder
    results_file = LOGS_DIR / 'all_results.csv'
    df_results.to_csv(results_file, index=False)

    print(f"\n{'='*60}")
    print(f"✅ RÉSULTATS EXTRAITS ET SAUVEGARDÉS")
    print(f"{'='*60}")
    print(f"\nFichier créé: {results_file}")
    print(f"Nombre d'entraînements: {len(all_results)}")
    print(f"\nVous pouvez maintenant lancer le notebook 05 ! 🎉")
    print(f"\nTableau récapitulatif:")
    print(df_results[['model', 'augmentation', 'val_dice', 'val_miou', 'val_accuracy']].to_string(index=False))


if __name__ == '__main__':
    extract_from_logs()
