"""
Data Preprocessing Pipeline for RGB-IR Pedestrian Detection.
Handles dataset loading, augmentation, and batch preparation for dual-modality inputs.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List, Dict, Optional, Callable
import numpy as np
import cv2
from pathlib import Path
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DualModalityPedestrianDataset(Dataset):
    """
    Dataset for loading aligned RGB and Thermal image pairs with pedestrian annotations.
    Supports various augmentation strategies for robust model training.
    """

    def __init__(self, rgb_root: str, thermal_root: str, annotation_file: str,
                 img_size: Tuple[int, int] = (640, 640),
                 augment: bool = True,
                 cache_images: bool = False,
                 mosaic_prob: float = 0.5,
                 mixup_prob: float = 0.15):
        """
        Args:
            rgb_root: Path to RGB images directory
            thermal_root: Path to thermal images directory
            annotation_file: Path to YOLO format annotations
            img_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            cache_images: Cache images in memory for faster loading
            mosaic_prob: Probability of applying mosaic augmentation
            mixup_prob: Probability of applying mixup augmentation
        """
        self.rgb_root = Path(rgb_root)
        self.thermal_root = Path(thermal_root)
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob if augment else 0.0
        self.mixup_prob = mixup_prob if augment else 0.0

        # Load image paths and labels
        self.samples = self._load_annotations(annotation_file)
        self.num_samples = len(self.samples)

        # Initialize cache
        self.cache_images = cache_images
        self.cached_data = {} if cache_images else None

        # Build augmentation pipeline
        self.transforms = self._build_transforms(augment)

    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        """Load YOLO format annotations."""
        samples = []

        # Implementation depends on your annotation format
        # This is a placeholder structure
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_path = parts[0]

                # Extract labels: class, x_center, y_center, width, height
                labels = []
                for i in range(1, len(parts), 5):
                    if i + 4 < len(parts):
                        labels.append([
                            float(parts[i]),      # class
                            float(parts[i+1]),    # x_center
                            float(parts[i+2]),    # y_center
                            float(parts[i+3]),    # width
                            float(parts[i+4])     # height
                        ])

                samples.append({
                    'rgb_path': self.rgb_root / img_path,
                    'thermal_path': self.thermal_root / img_path,
                    'labels': np.array(labels, dtype=np.float32)
                })

        return samples

    def _build_transforms(self, augment: bool) -> A.Compose:
        """Build augmentation pipeline using albumentations."""
        if augment:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Blur(blur_limit=3, p=0.1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))
        else:
            return A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            img_pair: Concatenated RGB and thermal images [6, H, W]
            labels: Bounding box annotations [N, 6] (index, class, x, y, w, h)
            path: Image path string
        """
        # Apply mosaic augmentation with probability
        if self.augment and random.random() < self.mosaic_prob:
            return self._load_mosaic(index)

        # Load single sample
        sample = self.samples[index]

        # Load from cache or disk
        if self.cache_images and index in self.cached_data:
            rgb_img, thermal_img, labels = self.cached_data[index]
        else:
            rgb_img = cv2.imread(str(sample['rgb_path']))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            thermal_img = cv2.imread(str(sample['thermal_path']))
            if thermal_img.shape[2] == 3:
                thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
                thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2RGB)

            labels = sample['labels'].copy()

            if self.cache_images:
                self.cached_data[index] = (rgb_img, thermal_img, labels)

        # Apply transformations
        if len(labels) > 0:
            bboxes = labels[:, 1:5].tolist()
            class_labels = labels[:, 0].tolist()

            # Transform RGB
            transformed_rgb = self.transforms(
                image=rgb_img,
                bboxes=bboxes,
                class_labels=class_labels
            )

            # Transform thermal with same parameters for consistency
            transformed_thermal = self.transforms(
                image=thermal_img,
                bboxes=bboxes,
                class_labels=class_labels
            )

            rgb_img = transformed_rgb['image']
            thermal_img = transformed_thermal['image']

            # Update labels after transformation
            if len(transformed_rgb['bboxes']) > 0:
                new_labels = []
                for bbox, cls in zip(transformed_rgb['bboxes'], transformed_rgb['class_labels']):
                    new_labels.append([cls] + list(bbox))
                labels = np.array(new_labels, dtype=np.float32)
            else:
                labels = np.zeros((0, 5), dtype=np.float32)

        else:
            # No labels, just transform images
            rgb_img = self.transforms(image=rgb_img)['image']
            thermal_img = self.transforms(image=thermal_img)['image']
            labels = np.zeros((0, 5), dtype=np.float32)

        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
        thermal_tensor = torch.from_numpy(thermal_img).permute(2, 0, 1).float() / 255.0

        # Concatenate RGB and thermal
        img_pair = torch.cat([rgb_tensor, thermal_tensor], dim=0)

        # Format labels: add image index column
        labels_out = torch.zeros((len(labels), 6))
        if len(labels) > 0:
            labels_out[:, 1:] = torch.from_numpy(labels)

        return img_pair, labels_out, str(sample['rgb_path'])

    def _load_mosaic(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load 4 images and create a mosaic augmentation.
        Effective for learning multi-scale pedestrian features.
        """
        indices = [index] + random.choices(range(self.num_samples), k=3)

        mosaic_h, mosaic_w = self.img_size[0] * 2, self.img_size[1] * 2
        mosaic_rgb = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        mosaic_thermal = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        mosaic_labels = []

        # Center point of mosaic
        center_x, center_y = mosaic_w // 2, mosaic_h // 2

        for i, idx in enumerate(indices):
            sample = self.samples[idx]

            # Load images
            rgb_img = cv2.imread(str(sample['rgb_path']))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            thermal_img = cv2.imread(str(sample['thermal_path']))
            if thermal_img.shape[2] == 3:
                thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
                thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2RGB)

            h, w = rgb_img.shape[:2]
            labels = sample['labels'].copy()

            # Determine placement in mosaic
            if i == 0:  # Top-left
                x1, y1, x2, y2 = 0, 0, center_x, center_y
            elif i == 1:  # Top-right
                x1, y1, x2, y2 = center_x, 0, mosaic_w, center_y
            elif i == 2:  # Bottom-left
                x1, y1, x2, y2 = 0, center_y, center_x, mosaic_h
            else:  # Bottom-right
                x1, y1, x2, y2 = center_x, center_y, mosaic_w, mosaic_h

            # Resize image to fit quadrant
            quad_h, quad_w = y2 - y1, x2 - x1
            rgb_resized = cv2.resize(rgb_img, (quad_w, quad_h))
            thermal_resized = cv2.resize(thermal_img, (quad_w, quad_h))

            # Place in mosaic
            mosaic_rgb[y1:y2, x1:x2] = rgb_resized
            mosaic_thermal[y1:y2, x1:x2] = thermal_resized

            # Adjust labels
            if len(labels) > 0:
                for label in labels:
                    cls, x_c, y_c, w_l, h_l = label
                    # Convert from relative to absolute coordinates
                    x_c_abs = x_c * quad_w + x1
                    y_c_abs = y_c * quad_h + y1
                    w_abs = w_l * quad_w
                    h_abs = h_l * quad_h

                    # Convert back to relative coordinates for mosaic
                    x_c_rel = x_c_abs / mosaic_w
                    y_c_rel = y_c_abs / mosaic_h
                    w_rel = w_abs / mosaic_w
                    h_rel = h_abs / mosaic_h

                    mosaic_labels.append([cls, x_c_rel, y_c_rel, w_rel, h_rel])

        # Resize mosaic to target size
        mosaic_rgb = cv2.resize(mosaic_rgb, (self.img_size[1], self.img_size[0]))
        mosaic_thermal = cv2.resize(mosaic_thermal, (self.img_size[1], self.img_size[0]))

        # Convert to tensors
        rgb_tensor = torch.from_numpy(mosaic_rgb).permute(2, 0, 1).float() / 255.0
        thermal_tensor = torch.from_numpy(mosaic_thermal).permute(2, 0, 1).float() / 255.0
        img_pair = torch.cat([rgb_tensor, thermal_tensor], dim=0)

        # Format labels
        labels_out = torch.zeros((len(mosaic_labels), 6))
        if len(mosaic_labels) > 0:
            mosaic_labels = np.array(mosaic_labels, dtype=np.float32)
            labels_out = torch.zeros((len(mosaic_labels), 6))
            labels_out[:, 1:] = torch.from_numpy(mosaic_labels)

        return img_pair, labels_out, "mosaic"


class PedestrianBalancedSampler(Sampler):
    """
    Balanced sampler that ensures equal representation of different pedestrian scales.
    Helps model learn to detect pedestrians at various distances.
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.shuffle = shuffle

        # Categorize samples by pedestrian scale
        self.scale_indices = {'small': [], 'medium': [], 'large': []}
        self._categorize_by_scale()

    def _categorize_by_scale(self):
        """Categorize pedestrians into small, medium, large based on bbox size."""
        for idx in range(self.num_samples):
            sample = self.dataset.samples[idx]
            labels = sample['labels']

            if len(labels) == 0:
                continue

            # Compute average box area
            avg_area = np.mean(labels[:, 3] * labels[:, 4])  # width * height

            if avg_area < 0.02:  # Small pedestrians
                self.scale_indices['small'].append(idx)
            elif avg_area < 0.1:  # Medium pedestrians
                self.scale_indices['medium'].append(idx)
            else:  # Large pedestrians
                self.scale_indices['large'].append(idx)

    def __iter__(self):
        # Sample equally from each scale category
        indices = []

        num_per_category = self.num_samples // 3

        for category in ['small', 'medium', 'large']:
            cat_indices = self.scale_indices[category]
            if len(cat_indices) > 0:
                sampled = random.choices(cat_indices, k=num_per_category)
                indices.extend(sampled)

        if self.shuffle:
            random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples


def collate_dual_modality(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[str], List]:
    """
    Custom collate function for dual-modality batches.
    Handles variable number of pedestrians per image.
    """
    imgs, labels, paths, shapes = [], [], [], []

    for img, label, path in batch:
        imgs.append(img)
        labels.append(label)
        paths.append(path)

    # Stack images
    imgs = torch.stack(imgs, 0)

    # Assign batch indices to labels
    for i, label in enumerate(labels):
        label[:, 0] = i

    labels = torch.cat(labels, 0)

    return imgs, labels, paths, shapes


def create_pedestrian_dataloader(rgb_root: str, thermal_root: str,
                                 annotation_file: str,
                                 batch_size: int = 16,
                                 img_size: Tuple[int, int] = (640, 640),
                                 augment: bool = True,
                                 cache_images: bool = False,
                                 num_workers: int = 8,
                                 shuffle: bool = True,
                                 pin_memory: bool = True) -> DataLoader:
    """
    Create DataLoader for RGB-IR pedestrian detection.

    Args:
        rgb_root: Path to RGB images
        thermal_root: Path to thermal images
        annotation_file: Path to annotations
        batch_size: Batch size
        img_size: Target image size
        augment: Whether to apply augmentation
        cache_images: Cache images in memory
        num_workers: Number of dataloader workers
        shuffle: Shuffle data
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    dataset = DualModalityPedestrianDataset(
        rgb_root=rgb_root,
        thermal_root=thermal_root,
        annotation_file=annotation_file,
        img_size=img_size,
        augment=augment,
        cache_images=cache_images
    )

    sampler = PedestrianBalancedSampler(dataset, shuffle=shuffle) if shuffle else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dual_modality,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return loader


class ThermalImageNormalizer:
    """
    Normalizes thermal images to consistent temperature range.
    Handles different thermal camera calibrations.
    """

    def __init__(self, method: str = 'adaptive'):
        """
        Args:
            method: Normalization method ('adaptive', 'global', 'histogram')
        """
        self.method = method

    def __call__(self, thermal_img: np.ndarray) -> np.ndarray:
        if self.method == 'adaptive':
            return self._adaptive_normalize(thermal_img)
        elif self.method == 'global':
            return self._global_normalize(thermal_img)
        elif self.method == 'histogram':
            return self._histogram_equalize(thermal_img)
        else:
            return thermal_img

    def _adaptive_normalize(self, img: np.ndarray) -> np.ndarray:
        """Adaptive normalization based on image statistics."""
        mean = np.mean(img)
        std = np.std(img)

        normalized = (img - mean) / (std + 1e-6)
        normalized = np.clip(normalized, -3, 3)
        normalized = (normalized + 3) / 6  # Scale to [0, 1]

        return (normalized * 255).astype(np.uint8)

    def _global_normalize(self, img: np.ndarray) -> np.ndarray:
        """Global min-max normalization."""
        min_val = np.min(img)
        max_val = np.max(img)

        normalized = (img - min_val) / (max_val - min_val + 1e-6)

        return (normalized * 255).astype(np.uint8)

    def _histogram_equalize(self, img: np.ndarray) -> np.ndarray:
        """Histogram equalization for contrast enhancement."""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img

        equalized = cv2.equalizeHist(img_gray)

        if len(img.shape) == 3:
            equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

        return equalized
