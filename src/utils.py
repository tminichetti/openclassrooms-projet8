
"""Utilitaires pour le projet de segmentation Cityscapes."""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from pathlib import Path

# Configuration
IMG_HEIGHT = 256
IMG_WIDTH = 512
N_CLASSES = 8

CATEGORIES = {
    0: {'name': 'void', 'color': (0, 0, 0)},
    1: {'name': 'flat', 'color': (128, 64, 128)},
    2: {'name': 'construction', 'color': (70, 70, 70)},
    3: {'name': 'object', 'color': (153, 153, 153)},
    4: {'name': 'nature', 'color': (107, 142, 35)},
    5: {'name': 'sky', 'color': (70, 130, 180)},
    6: {'name': 'human', 'color': (220, 20, 60)},
    7: {'name': 'vehicle', 'color': (0, 0, 142)}
}

LABEL_TO_CATEGORY = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
    7: 1, 8: 1, 9: 1, 10: 1,
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,
    17: 3, 18: 3, 19: 3, 20: 3,
    21: 4, 22: 4,
    23: 5,
    24: 6, 25: 6,
    26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7,
}

# LUT pour conversion rapide
LUT = np.zeros(256, dtype=np.uint8)
for label_id, cat_id in LABEL_TO_CATEGORY.items():
    LUT[label_id] = cat_id


def load_image(path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Charge et preprocesse une image."""
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return img


def load_label(path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Charge et convertit un mask en 8 categories."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    mask = LUT[mask]
    return mask


def mask_to_onehot(mask, n_classes=N_CLASSES):
    """Convertit un mask en one-hot."""
    return np.eye(n_classes, dtype=np.float32)[mask]


def mask_to_rgb(mask):
    """Convertit un mask en image RGB coloree."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cat_id, info in CATEGORIES.items():
        rgb[mask == cat_id] = info['color']
    return rgb
