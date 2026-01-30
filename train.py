"""
Script d'entraînement pour les modèles de segmentation
Projet 8 - OpenClassrooms

Usage:
    python train.py --model unet --augmentation --epochs 30
    python train.py --model vgg16 --no-augmentation --epochs 20
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sans affichage
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger, TensorBoard
)

# Importer les utilitaires locaux
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils import load_image, load_label, mask_to_onehot, CATEGORIES, LUT
from models import build_unet, dice_coefficient, dice_loss, combined_loss


class CityscapesGenerator(keras.utils.Sequence):
    """Générateur de données Keras."""

    def __init__(self, data, batch_size, target_size, n_classes, augmentation=None, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_masks = []

        for idx in batch_indexes:
            sample = self.data[idx]
            image = load_image(sample['image'], self.target_size)
            mask = load_label(sample['label'], self.target_size)

            if self.augmentation:
                image, mask = self.augmentation(image, mask)

            batch_images.append(image)
            batch_masks.append(mask_to_onehot(mask, self.n_classes))

        return np.array(batch_images), np.array(batch_masks)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class DataAugmentation:
    """Augmentation cohérente image + mask."""

    def __init__(self):
        pass

    def __call__(self, image, mask):
        import cv2

        # Flip horizontal
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # Brightness
        factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 1)

        # Contrast
        factor = np.random.uniform(0.9, 1.1)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * factor + mean, 0, 1)

        return image.astype(np.float32), mask


class MeanIoU(tf.keras.metrics.Metric):
    """Mean Intersection over Union."""

    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)

        iou_per_class = []
        for c in range(self.num_classes):
            true_c = tf.cast(tf.equal(y_true_labels, c), tf.float32)
            pred_c = tf.cast(tf.equal(y_pred_labels, c), tf.float32)
            intersection = tf.reduce_sum(true_c * pred_c)
            union = tf.reduce_sum(true_c) + tf.reduce_sum(pred_c) - intersection
            iou = tf.where(union > 0, intersection / union, 0.0)
            iou_per_class.append(iou)

        mean_iou = tf.reduce_mean(tf.stack(iou_per_class))
        self.total_iou.assign_add(mean_iou)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_iou / self.count

    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)


def build_model(model_type, input_shape, n_classes):
    """Construit le modèle selon le type."""
    if model_type == 'unet':
        return build_unet(input_shape, n_classes, filters=[32, 64, 128, 256, 512])
    elif model_type == 'vgg16':
        return build_unet_vgg16(input_shape, n_classes)
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")


def build_unet_vgg16(input_shape, n_classes, freeze_encoder=True):
    """U-Net avec VGG16."""
    def conv_block(inputs, n_filters):
        x = layers.Conv2D(n_filters, 3, padding='same', kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(n_filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def decoder_block(inputs, skip, n_filters):
        x = layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
        x = layers.Concatenate()([x, skip])
        x = layers.Dropout(0.3)(x)
        x = conv_block(x, n_filters)
        return x

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    if freeze_encoder:
        for layer in base_model.layers:
            layer.trainable = False

    skip1 = base_model.get_layer('block1_conv2').output
    skip2 = base_model.get_layer('block2_conv2').output
    skip3 = base_model.get_layer('block3_conv3').output
    skip4 = base_model.get_layer('block4_conv3').output
    bottleneck = base_model.get_layer('block5_conv3').output

    dec4 = decoder_block(bottleneck, skip4, 512)
    dec3 = decoder_block(dec4, skip3, 256)
    dec2 = decoder_block(dec3, skip2, 128)
    dec1 = decoder_block(dec2, skip1, 64)

    outputs = layers.Conv2D(n_classes, (1, 1), activation='softmax')(dec1)

    return Model(base_model.input, outputs, name='UNet_VGG16')


def plot_history(history, save_path):
    """Génère les courbes d'apprentissage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history['loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Dice
    axes[1, 0].plot(history['dice_coefficient'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_dice_coefficient'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # IoU
    axes[1, 1].plot(history['mean_iou'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_mean_iou'], label='Validation', linewidth=2)
    axes[1, 1].set_title('Mean IoU')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Entraînement modèle de segmentation')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'vgg16'],
                        help='Type de modèle (unet ou vgg16)')
    parser.add_argument('--augmentation', action='store_true', default=False,
                        help='Activer data augmentation')
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false',
                        help='Désactiver data augmentation')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Nombre d\'epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Taille des batches')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience pour early stopping')

    args = parser.parse_args()

    # Configuration
    DATA_DIR = Path('data')
    MODELS_DIR = Path('models')
    LOGS_DIR = Path('logs')

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # Charger config
    with open(DATA_DIR / 'config.json', 'r') as f:
        config = json.load(f)

    IMG_HEIGHT = config['img_height']
    IMG_WIDTH = config['img_width']
    N_CLASSES = config['n_classes']
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

    # Nom de l'expérience
    aug_str = 'aug' if args.augmentation else 'no-aug'
    exp_name = f"{args.model}_{aug_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = LOGS_DIR / exp_name
    exp_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("ENTRAÎNEMENT MODÈLE DE SEGMENTATION")
    print("="*60)
    print(f"\nExpérience: {exp_name}")
    print(f"Modèle: {args.model}")
    print(f"Augmentation: {'Oui' if args.augmentation else 'Non'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Input shape: {INPUT_SHAPE}")

    # GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU: {len(gpus)} disponible(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("GPU: Aucun (CPU uniquement)")

    # Charger données
    print("\nChargement des données...")
    train_df = pd.read_csv(DATA_DIR / 'train_paths.csv')
    val_df = pd.read_csv(DATA_DIR / 'val_paths.csv')

    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')

    print(f"  Train: {len(train_data)} images")
    print(f"  Val: {len(val_data)} images")

    # Créer générateurs
    augmentation = DataAugmentation() if args.augmentation else None

    train_gen = CityscapesGenerator(
        train_data, args.batch_size, (IMG_HEIGHT, IMG_WIDTH),
        N_CLASSES, augmentation=augmentation, shuffle=True
    )

    val_gen = CityscapesGenerator(
        val_data, args.batch_size, (IMG_HEIGHT, IMG_WIDTH),
        N_CLASSES, augmentation=None, shuffle=False
    )

    print(f"  Train batches: {len(train_gen)}")
    print(f"  Val batches: {len(val_gen)}")

    # Créer modèle
    print(f"\nConstruction du modèle {args.model}...")
    model = build_model(args.model, INPUT_SHAPE, N_CLASSES)
    print(f"  Paramètres: {model.count_params():,}")

    # Compiler
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient, MeanIoU(num_classes=N_CLASSES)]
    )

    # Callbacks
    model_path = MODELS_DIR / f'{exp_name}_best.keras'

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=args.patience // 2,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(
            filename=str(exp_dir / 'training_log.csv'),
            separator=',',
            append=False
        ),
        TensorBoard(log_dir=str(exp_dir))
    ]

    # Sauvegarder config expérience
    exp_config = {
        'experiment_name': exp_name,
        'model': args.model,
        'augmentation': args.augmentation,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'input_shape': INPUT_SHAPE,
        'n_classes': N_CLASSES,
        'train_samples': len(train_data),
        'val_samples': len(val_data)
    }

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(exp_config, f, indent=2)

    # Entraînement
    print("\n" + "="*60)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*60 + "\n")

    start_time = time.time()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    elapsed_time = time.time() - start_time

    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print(f"\nTemps total: {elapsed_time/60:.1f} minutes")

    # Sauvegarder historique
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(exp_dir / 'history.csv', index=False)

    # Courbes
    plot_history(history.history, exp_dir / 'training_curves.png')

    # Résultats
    best_epoch = np.argmin(history.history['val_loss'])
    best_results = {
        'experiment': exp_name,
        'model': args.model,
        'augmentation': args.augmentation,
        'epochs_trained': len(history.history['loss']),
        'best_epoch': best_epoch + 1,
        'training_time_minutes': elapsed_time / 60,
        'val_loss': float(history.history['val_loss'][best_epoch]),
        'val_accuracy': float(history.history['val_accuracy'][best_epoch]),
        'val_dice': float(history.history['val_dice_coefficient'][best_epoch]),
        'val_miou': float(history.history['val_mean_iou'][best_epoch]),
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat()
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(best_results, f, indent=2)

    # Afficher résultats
    print(f"\nMeilleurs scores (epoch {best_results['best_epoch']}):")
    print(f"  - Loss: {best_results['val_loss']:.4f}")
    print(f"  - Accuracy: {best_results['val_accuracy']:.4f}")
    print(f"  - Dice: {best_results['val_dice']:.4f}")
    print(f"  - mIoU: {best_results['val_miou']:.4f}")

    print(f"\nModèle sauvegardé: {model_path}")
    print(f"Logs: {exp_dir}")

    # Ajouter au registre global
    results_file = LOGS_DIR / 'all_results.csv'
    if results_file.exists():
        df = pd.read_csv(results_file)
        df = pd.concat([df, pd.DataFrame([best_results])], ignore_index=True)
    else:
        df = pd.DataFrame([best_results])
    df.to_csv(results_file, index=False)

    print(f"\nRésultats ajoutés à: {results_file}")
    print("\n✅ Entraînement terminé avec succès!\n")


if __name__ == '__main__':
    main()
