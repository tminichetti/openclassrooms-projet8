# Cellule à ajouter à la fin du Notebook 04

Ajouter cette cellule **après la cellule 31** (dernière cellule actuelle) du notebook `04_training.ipynb`.

## Nouvelle Cellule (Markdown)

```markdown
## 12. Export des Résultats pour Comparaison

Sauvegarde des résultats dans un fichier consolidé pour analyse dans le notebook 05.
```

## Nouvelle Cellule (Code)

```python
# Calculer le temps d'entraînement total
import time

# Si vous n'avez pas mesuré le temps, estimer depuis l'historique
# Sinon, utiliser la variable start_time/end_time si vous les avez définies

# Extraire les meilleurs résultats
best_epoch = np.argmin(history.history['val_loss'])

# Créer le dictionnaire de résultats
results_entry = {
    'experiment': EXPERIMENT_NAME,
    'model': 'unet',  # Modifier si vous utilisez un autre modèle (vgg16, etc.)
    'augmentation': True,  # Modifier selon si vous avez utilisé l'augmentation
    'epochs_trained': len(history.history['loss']),
    'best_epoch': best_epoch + 1,
    'training_time_minutes': 0,  # À ajuster si vous avez mesuré le temps
    'val_loss': float(history.history['val_loss'][best_epoch]),
    'val_accuracy': float(history.history['val_accuracy'][best_epoch]),
    'val_dice': float(history.history['val_dice_coefficient'][best_epoch]),
    'val_miou': float(history.history['val_mean_iou'][best_epoch]),
    'model_path': str(MODELS_DIR / 'unet_best.keras'),
    'timestamp': datetime.now().isoformat()
}

# Charger ou créer le fichier de résultats consolidés
results_file = LOGS_DIR / 'all_results.csv'

if results_file.exists():
    # Charger les résultats existants
    df_results = pd.read_csv(results_file)
    # Ajouter le nouveau résultat
    df_results = pd.concat([df_results, pd.DataFrame([results_entry])], ignore_index=True)
else:
    # Créer un nouveau DataFrame
    df_results = pd.DataFrame([results_entry])

# Sauvegarder
df_results.to_csv(results_file, index=False)

print("\n" + "="*60)
print("RÉSULTATS EXPORTÉS POUR COMPARAISON")
print("="*60)
print(f"\n✅ Résultats ajoutés à: {results_file}")
print(f"\nVous pouvez maintenant lancer le notebook 05 pour comparer les modèles !")
print("\nRésumé de cet entraînement:")
print(f"  - Modèle: {results_entry['model']}")
print(f"  - Augmentation: {results_entry['augmentation']}")
print(f"  - Dice: {results_entry['val_dice']:.4f}")
print(f"  - mIoU: {results_entry['val_miou']:.4f}")
print(f"  - Accuracy: {results_entry['val_accuracy']:.4f}")
```

## Instructions

1. **Ouvrir le notebook 04** : `notebooks/04_training.ipynb`

2. **Aller à la fin du notebook** (après la cellule 31)

3. **Ajouter une nouvelle cellule Markdown** avec le titre :
   ```
   ## 12. Export des Résultats pour Comparaison
   ```

4. **Ajouter une nouvelle cellule Code** avec le code ci-dessus

5. **Modifier les valeurs** selon votre entraînement :
   - `'model': 'unet'` → Changer en `'vgg16'` si vous utilisez VGG16
   - `'augmentation': True` → Mettre `False` si pas d'augmentation
   - `'training_time_minutes': 0` → Mettre le temps réel si vous l'avez mesuré

6. **Exécuter la cellule** après chaque entraînement

## Alternative : Mesurer le temps automatiquement

Si vous voulez mesurer le temps automatiquement, ajoutez **avant** l'entraînement (cellule 22) :

```python
import time
start_time = time.time()
```

Et **après** l'entraînement (juste après la cellule 22), ajoutez :

```python
elapsed_time = time.time() - start_time
training_time_minutes = elapsed_time / 60
print(f"\nTemps total: {training_time_minutes:.1f} minutes")
```

Puis dans la dernière cellule, remplacer :
```python
'training_time_minutes': 0,
```

Par :
```python
'training_time_minutes': training_time_minutes,
```

## Après ça

Une fois cette cellule ajoutée et exécutée, le notebook 05 fonctionnera parfaitement ! 🎉
