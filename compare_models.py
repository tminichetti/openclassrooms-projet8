"""
Script de comparaison des modèles entraînés
Génère un tableau comparatif pour la note technique

Usage:
    python compare_models.py
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """Charge tous les résultats d'entraînement."""
    results_file = Path('logs/all_results.csv')

    if not results_file.exists():
        print("❌ Aucun résultat trouvé")
        print("Lancez d'abord des entraînements avec train.py")
        return None

    df = pd.read_csv(results_file)
    return df

def print_comparison_table(df):
    """Affiche un tableau comparatif formaté."""
    print("\n" + "="*100)
    print("TABLEAU COMPARATIF DES MODÈLES")
    print("="*100)

    # Colonnes à afficher
    cols = ['model', 'augmentation', 'val_dice', 'val_miou', 'val_accuracy',
            'training_time_minutes', 'epochs_trained']

    display_df = df[cols].copy()
    display_df.columns = ['Modèle', 'Augmentation', 'Dice', 'mIoU', 'Accuracy',
                          'Temps (min)', 'Epochs']

    # Formatter
    display_df['Dice'] = display_df['Dice'].apply(lambda x: f"{x:.4f}")
    display_df['mIoU'] = display_df['mIoU'].apply(lambda x: f"{x:.4f}")
    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.4f}")
    display_df['Temps (min)'] = display_df['Temps (min)'].apply(lambda x: f"{x:.1f}")
    display_df['Augmentation'] = display_df['Augmentation'].apply(lambda x: 'Oui' if x else 'Non')

    print(display_df.to_string(index=False))
    print("="*100 + "\n")

def analyze_augmentation_impact(df):
    """Analyse l'impact de l'augmentation."""
    print("\n" + "="*100)
    print("ANALYSE DE L'IMPACT DE L'AUGMENTATION")
    print("="*100 + "\n")

    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        if len(model_df) < 2:
            continue

        with_aug = model_df[model_df['augmentation'] == True]
        without_aug = model_df[model_df['augmentation'] == False]

        if len(with_aug) == 0 or len(without_aug) == 0:
            continue

        print(f"Modèle: {model.upper()}")
        print("-" * 60)

        metrics = ['val_dice', 'val_miou', 'val_accuracy']
        metric_names = ['Dice', 'mIoU', 'Accuracy']

        for metric, name in zip(metrics, metric_names):
            val_with = with_aug[metric].values[0]
            val_without = without_aug[metric].values[0]
            gain = ((val_with - val_without) / val_without) * 100

            print(f"  {name:10} - Sans aug: {val_without:.4f} | Avec aug: {val_with:.4f} | Gain: {gain:+.2f}%")

        print()

def find_best_model(df):
    """Trouve le meilleur modèle."""
    print("\n" + "="*100)
    print("MEILLEUR MODÈLE")
    print("="*100 + "\n")

    # Trier par Dice (métrique principale)
    best = df.loc[df['val_dice'].idxmax()]

    print(f"🏆 Modèle sélectionné: {best['model'].upper()}")
    print(f"   Augmentation: {'Oui' if best['augmentation'] else 'Non'}")
    print(f"   Dice: {best['val_dice']:.4f}")
    print(f"   mIoU: {best['val_miou']:.4f}")
    print(f"   Accuracy: {best['val_accuracy']:.4f}")
    print(f"   Temps d'entraînement: {best['training_time_minutes']:.1f} min")
    print(f"   Epochs: {best['epochs_trained']}")
    print(f"\n   Modèle sauvegardé: {best['model_path']}")

    return best

def plot_comparison(df, save_path='logs/comparison.png'):
    """Génère des graphiques de comparaison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Préparer les données
    df_plot = df.copy()
    df_plot['model_aug'] = df_plot.apply(
        lambda x: f"{x['model'].upper()}\n{'avec aug' if x['augmentation'] else 'sans aug'}",
        axis=1
    )

    # Dice
    axes[0, 0].bar(range(len(df_plot)), df_plot['val_dice'], color=['#2ecc71' if x else '#e74c3c' for x in df_plot['augmentation']])
    axes[0, 0].set_xticks(range(len(df_plot)))
    axes[0, 0].set_xticklabels(df_plot['model_aug'], fontsize=9)
    axes[0, 0].set_title('Dice Coefficient', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Dice')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # mIoU
    axes[0, 1].bar(range(len(df_plot)), df_plot['val_miou'], color=['#2ecc71' if x else '#e74c3c' for x in df_plot['augmentation']])
    axes[0, 1].set_xticks(range(len(df_plot)))
    axes[0, 1].set_xticklabels(df_plot['model_aug'], fontsize=9)
    axes[0, 1].set_title('Mean IoU', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Accuracy
    axes[1, 0].bar(range(len(df_plot)), df_plot['val_accuracy'], color=['#2ecc71' if x else '#e74c3c' for x in df_plot['augmentation']])
    axes[1, 0].set_xticks(range(len(df_plot)))
    axes[1, 0].set_xticklabels(df_plot['model_aug'], fontsize=9)
    axes[1, 0].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Temps
    axes[1, 1].bar(range(len(df_plot)), df_plot['training_time_minutes'], color=['#3498db' for _ in df_plot['augmentation']])
    axes[1, 1].set_xticks(range(len(df_plot)))
    axes[1, 1].set_xticklabels(df_plot['model_aug'], fontsize=9)
    axes[1, 1].set_title('Temps d\'entraînement', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Minutes')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Avec augmentation'),
        Patch(facecolor='#e74c3c', label='Sans augmentation')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé: {save_path}")
    plt.close()

def export_latex_table(df, save_path='logs/comparison_table.tex'):
    """Exporte le tableau en format LaTeX pour la note technique."""
    df_export = df[['model', 'augmentation', 'val_dice', 'val_miou',
                    'val_accuracy', 'training_time_minutes', 'epochs_trained']].copy()

    df_export.columns = ['Modèle', 'Augmentation', 'Dice', 'mIoU', 'Accuracy',
                         'Temps (min)', 'Epochs']

    df_export['Augmentation'] = df_export['Augmentation'].apply(lambda x: 'Oui' if x else 'Non')
    df_export['Modèle'] = df_export['Modèle'].str.upper()

    latex_table = df_export.to_latex(index=False, float_format="%.4f",
                                     caption="Comparaison des modèles de segmentation",
                                     label="tab:model_comparison")

    with open(save_path, 'w') as f:
        f.write(latex_table)

    print(f"📄 Tableau LaTeX sauvegardé: {save_path}")

def export_markdown_table(df, save_path='logs/comparison_table.md'):
    """Exporte le tableau en format Markdown."""
    df_export = df[['model', 'augmentation', 'val_dice', 'val_miou',
                    'val_accuracy', 'training_time_minutes', 'epochs_trained']].copy()

    df_export.columns = ['Modèle', 'Augmentation', 'Dice', 'mIoU', 'Accuracy',
                         'Temps (min)', 'Epochs']

    df_export['Augmentation'] = df_export['Augmentation'].apply(lambda x: 'Oui' if x else 'Non')
    df_export['Modèle'] = df_export['Modèle'].str.upper()
    df_export['Dice'] = df_export['Dice'].apply(lambda x: f"{x:.4f}")
    df_export['mIoU'] = df_export['mIoU'].apply(lambda x: f"{x:.4f}")
    df_export['Accuracy'] = df_export['Accuracy'].apply(lambda x: f"{x:.4f}")
    df_export['Temps (min)'] = df_export['Temps (min)'].apply(lambda x: f"{x:.1f}")

    with open(save_path, 'w') as f:
        f.write("# Comparaison des Modèles de Segmentation\n\n")
        f.write(df_export.to_markdown(index=False))
        f.write("\n")

    print(f"📝 Tableau Markdown sauvegardé: {save_path}")

def main():
    print("\n" + "="*100)
    print("ANALYSE COMPARATIVE DES MODÈLES")
    print("="*100)

    # Charger résultats
    df = load_results()

    if df is None or len(df) == 0:
        return

    print(f"\n✅ {len(df)} entraînement(s) trouvé(s)")

    # Afficher tableau
    print_comparison_table(df)

    # Analyser augmentation
    analyze_augmentation_impact(df)

    # Meilleur modèle
    best = find_best_model(df)

    # Graphiques
    plot_comparison(df)

    # Export
    export_markdown_table(df)
    export_latex_table(df)

    print("\n" + "="*100)
    print("ANALYSE TERMINÉE")
    print("="*100)

    # Recommandations
    print("\n📋 PROCHAINES ÉTAPES:")
    print("  1. Copier le meilleur modèle dans api/model/:")
    print(f"     cp {best['model_path']} api/model/segmentation_model.h5")
    print("  2. Tester l'API localement")
    print("  3. Déployer sur Heroku")
    print("  4. Intégrer les résultats dans la note technique")
    print()

if __name__ == '__main__':
    main()
