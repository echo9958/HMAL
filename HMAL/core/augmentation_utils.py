"""
Advanced Augmentation Utilities for RGB-IR Pedestrian Detection.
Implements synchronized augmentations for dual-modality inputs.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional, Dict
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DualModalityAugmentor:
    """
    Synchronized augmentation pipeline for RGB and thermal image pairs.
    Ensures geometric transformations are applied consistently across modalities.
    """

    def __init__(self, img_size: int = 640, augment_level: str = 'medium'):
        """
        Args:
            img_size: Target image size
            augment_level: Augmentation intensity ('light', 'medium', 'heavy')
        """
        self.img_size = img_size
        self.augment_level = augment_level

        self.geometric_transforms = self._build_geometric_pipeline()
        self.rgb_photometric_transforms = self._build_rgb_photometric_pipeline()
        self.thermal_photometric_transforms = self._build_thermal_photometric_pipeline()

    def _build_geometric_pipeline(self) -> A.Compose:
        """Build geometric transformation pipeline (synchronized across modalities)."""
        if self.augment_level == 'light':
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    p=0.3
                ),
            ]
        elif self.augment_level == 'medium':
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5
                ),
                A.RandomResizedCrop(
                    height=self.img_size,
                    width=self.img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.3
                ),
            ]
        else:  # heavy
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=0.7
                ),
                A.RandomResizedCrop(
                    height=self.img_size,
                    width=self.img_size,
                    scale=(0.7, 1.0),
                    ratio=(0.85, 1.15),
                    p=0.4
                ),
                A.Perspective(scale=(0.02, 0.05), p=0.2),
            ]

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    def _build_rgb_photometric_pipeline(self) -> A.Compose:
        """Build RGB-specific photometric augmentations."""
        if self.augment_level == 'light':
            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.2
                ),
            ]
        elif self.augment_level == 'medium':
            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=0.3
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10, 30), p=0.2),
            ]
        else:  # heavy
            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.4
                ),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.15,
                    p=0.4
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.MotionBlur(blur_limit=(3, 5), p=0.2),
            ]

        return A.Compose(transforms)

    def _build_thermal_photometric_pipeline(self) -> A.Compose:
        """Build thermal-specific photometric augmentations."""
        if self.augment_level == 'light':
            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
            ]
        elif self.augment_level == 'medium':
            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(5, 20), p=0.2),
            ]
        else:  # heavy
            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.GaussNoise(var_limit=(5, 30), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ]

        return A.Compose(transforms)

    def __call__(self, rgb_image: np.ndarray, thermal_image: np.ndarray,
                 bboxes: List, class_labels: List) -> Tuple[np.ndarray, np.ndarray, List, List]:
        """
        Apply synchronized augmentations to RGB-thermal pair.

        Args:
            rgb_image: RGB image array [H, W, 3]
            thermal_image: Thermal image array [H, W, 3]
            bboxes: Bounding boxes in YOLO format
            class_labels: Class labels for each bbox

        Returns:
            Tuple of (augmented_rgb, augmented_thermal, augmented_bboxes, augmented_labels)
        """
        # Apply geometric transformations (synchronized)
        geometric_result = self.geometric_transforms(
            image=rgb_image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        rgb_geom = geometric_result['image']
        thermal_geom = self.geometric_transforms(
            image=thermal_image,
            bboxes=bboxes,
            class_labels=class_labels
        )['image']

        aug_bboxes = geometric_result['bboxes']
        aug_labels = geometric_result['class_labels']

        # Apply photometric transformations (independent)
        rgb_final = self.rgb_photometric_transforms(image=rgb_geom)['image']
        thermal_final = self.thermal_photometric_transforms(image=thermal_geom)['image']

        return rgb_final, thermal_final, aug_bboxes, aug_labels


class MosaicAugmentation:
    """
    Mosaic augmentation that combines 4 images into a grid.
    Effective for learning multi-scale pedestrian features.
    """

    def __init__(self, img_size: int = 640):
        self.img_size = img_size

    def __call__(self, images_rgb: List[np.ndarray], images_thermal: List[np.ndarray],
                 bboxes_list: List[List], labels_list: List[List]) -> Tuple:
        """
        Create mosaic from 4 image pairs.

        Args:
            images_rgb: List of 4 RGB images
            images_thermal: List of 4 thermal images
            bboxes_list: List of 4 bbox lists
            labels_list: List of 4 label lists

        Returns:
            Tuple of (mosaic_rgb, mosaic_thermal, mosaic_bboxes, mosaic_labels)
        """
        assert len(images_rgb) == 4, "Mosaic requires exactly 4 images"

        # Create empty mosaic canvases
        mosaic_h, mosaic_w = self.img_size * 2, self.img_size * 2
        mosaic_rgb = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        mosaic_thermal = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

        mosaic_bboxes = []
        mosaic_labels = []

        # Center point
        center_x, center_y = self.img_size, self.img_size

        for i, (rgb_img, thermal_img, bboxes, labels) in enumerate(
            zip(images_rgb, images_thermal, bboxes_list, labels_list)
        ):
            h, w = rgb_img.shape[:2]

            # Determine placement
            if i == 0:  # Top-left
                x1, y1, x2, y2 = 0, 0, center_x, center_y
            elif i == 1:  # Top-right
                x1, y1, x2, y2 = center_x, 0, mosaic_w, center_y
            elif i == 2:  # Bottom-left
                x1, y1, x2, y2 = 0, center_y, center_x, mosaic_h
            else:  # Bottom-right
                x1, y1, x2, y2 = center_x, center_y, mosaic_w, mosaic_h

            # Resize to fit quadrant
            quad_h, quad_w = y2 - y1, x2 - x1
            rgb_resized = cv2.resize(rgb_img, (quad_w, quad_h))
            thermal_resized = cv2.resize(thermal_img, (quad_w, quad_h))

            # Place in mosaic
            mosaic_rgb[y1:y2, x1:x2] = rgb_resized
            mosaic_thermal[y1:y2, x1:x2] = thermal_resized

            # Adjust bounding boxes
            for bbox, label in zip(bboxes, labels):
                xc, yc, bw, bh = bbox

                # Convert to absolute coordinates in quadrant
                xc_abs = xc * quad_w + x1
                yc_abs = yc * quad_h + y1
                bw_abs = bw * quad_w
                bh_abs = bh * quad_h

                # Convert to relative coordinates in mosaic
                xc_mosaic = xc_abs / mosaic_w
                yc_mosaic = yc_abs / mosaic_h
                bw_mosaic = bw_abs / mosaic_w
                bh_mosaic = bh_abs / mosaic_h

                # Clip to valid range
                xc_mosaic = np.clip(xc_mosaic, 0, 1)
                yc_mosaic = np.clip(yc_mosaic, 0, 1)
                bw_mosaic = np.clip(bw_mosaic, 0, 1)
                bh_mosaic = np.clip(bh_mosaic, 0, 1)

                mosaic_bboxes.append([xc_mosaic, yc_mosaic, bw_mosaic, bh_mosaic])
                mosaic_labels.append(label)

        # Resize mosaic to target size
        mosaic_rgb = cv2.resize(mosaic_rgb, (self.img_size, self.img_size))
        mosaic_thermal = cv2.resize(mosaic_thermal, (self.img_size, self.img_size))

        return mosaic_rgb, mosaic_thermal, mosaic_bboxes, mosaic_labels


class MixupAugmentation:
    """
    Mixup augmentation for dual-modality inputs.
    Blends two image pairs and their labels.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def __call__(self, rgb1: np.ndarray, thermal1: np.ndarray, labels1: np.ndarray,
                 rgb2: np.ndarray, thermal2: np.ndarray, labels2: np.ndarray) -> Tuple:
        """
        Mix two image pairs.

        Args:
            rgb1, thermal1, labels1: First image pair and labels
            rgb2, thermal2, labels2: Second image pair and labels

        Returns:
            Tuple of (mixed_rgb, mixed_thermal, mixed_labels)
        """
        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix images
        mixed_rgb = (lam * rgb1 + (1 - lam) * rgb2).astype(np.uint8)
        mixed_thermal = (lam * thermal1 + (1 - lam) * thermal2).astype(np.uint8)

        # Combine labels
        mixed_labels = np.vstack([labels1, labels2])

        return mixed_rgb, mixed_thermal, mixed_labels


class CutoutAugmentation:
    """
    Random cutout augmentation (erase random patches).
    Improves robustness to occlusions.
    """

    def __init__(self, num_holes: int = 1, max_h_size: int = 50, max_w_size: int = 50):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def __call__(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> Tuple:
        """
        Apply cutout to both modalities.

        Args:
            rgb_image: RGB image
            thermal_image: Thermal image

        Returns:
            Tuple of (cutout_rgb, cutout_thermal)
        """
        h, w = rgb_image.shape[:2]

        for _ in range(self.num_holes):
            # Sample hole size
            hole_h = random.randint(1, self.max_h_size)
            hole_w = random.randint(1, self.max_w_size)

            # Sample hole position
            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)

            # Apply cutout
            rgb_image[y:y+hole_h, x:x+hole_w] = 0
            thermal_image[y:y+hole_h, x:x+hole_w] = 0

        return rgb_image, thermal_image


class WeatherAugmentation:
    """
    Simulates various weather conditions for robustness.
    Includes rain, fog, snow effects.
    """

    def __init__(self):
        self.effects = ['rain', 'fog', 'snow', 'none']

    def add_rain(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add rain effect to image."""
        h, w = image.shape[:2]

        num_drops = int(intensity * 100)

        for _ in range(num_drops):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            length = random.randint(5, 15)

            # Draw raindrop
            cv2.line(image, (x, y), (x, min(y+length, h-1)), (200, 200, 200), 1)

        return image

    def add_fog(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add fog effect to image."""
        fog = np.ones_like(image) * 255

        alpha = intensity * 0.5
        fogged = cv2.addWeighted(image, 1-alpha, fog, alpha, 0)

        return fogged.astype(np.uint8)

    def add_snow(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add snow effect to image."""
        h, w = image.shape[:2]

        num_flakes = int(intensity * 200)

        for _ in range(num_flakes):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            size = random.randint(1, 3)

            cv2.circle(image, (x, y), size, (255, 255, 255), -1)

        return image

    def __call__(self, rgb_image: np.ndarray, effect: Optional[str] = None) -> np.ndarray:
        """
        Apply random weather effect to RGB image.
        Thermal images are typically unaffected by weather.

        Args:
            rgb_image: RGB image
            effect: Specific effect or None for random

        Returns:
            Augmented RGB image
        """
        if effect is None:
            effect = random.choice(self.effects)

        if effect == 'rain':
            rgb_image = self.add_rain(rgb_image, intensity=random.uniform(0.3, 0.7))
        elif effect == 'fog':
            rgb_image = self.add_fog(rgb_image, intensity=random.uniform(0.3, 0.6))
        elif effect == 'snow':
            rgb_image = self.add_snow(rgb_image, intensity=random.uniform(0.3, 0.7))

        return rgb_image


class ModalityDropout:
    """
    Randomly drop one modality during training.
    Encourages each modality to be independently informative.
    """

    def __init__(self, dropout_prob: float = 0.1):
        self.dropout_prob = dropout_prob

    def __call__(self, rgb_image: np.ndarray, thermal_image: np.ndarray) -> Tuple:
        """
        Randomly zero out one modality.

        Args:
            rgb_image: RGB image
            thermal_image: Thermal image

        Returns:
            Tuple of (rgb_image, thermal_image) with possible dropout
        """
        if random.random() < self.dropout_prob:
            if random.random() < 0.5:
                # Drop RGB
                rgb_image = np.zeros_like(rgb_image)
            else:
                # Drop thermal
                thermal_image = np.zeros_like(thermal_image)

        return rgb_image, thermal_image
