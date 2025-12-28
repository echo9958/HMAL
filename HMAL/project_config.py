"""
Project Configuration and Setup Guide for VI Pedestrian Detection System.
Contains default configurations, hyperparameters, and setup instructions.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


# Default Hyperparameters for Training
DEFAULT_HYPERPARAMETERS = {
    # Learning rate settings
    'lr0': 0.001,                    # initial learning rate
    'lrf': 0.01,                     # final OneCycleLR learning rate (lr0 * lrf)
    'momentum': 0.937,               # SGD momentum / Adam beta1
    'weight_decay': 0.0005,          # optimizer weight decay
    'warmup_epochs': 3.0,            # warmup epochs (fractions ok)
    'warmup_momentum': 0.8,          # warmup initial momentum
    'warmup_bias_lr': 0.1,           # warmup initial bias lr

    # Loss weights
    'box': 0.05,                     # box loss gain
    'cls': 0.5,                      # cls loss gain
    'cls_pw': 1.0,                   # cls BCELoss positive_weight
    'obj': 1.0,                      # obj loss gain (scale with pixels)
    'obj_pw': 1.0,                   # obj BCELoss positive_weight
    'iou_t': 0.20,                   # IoU training threshold
    'anchor_t': 4.0,                 # anchor-multiple threshold

    # Augmentation parameters
    'hsv_h': 0.015,                  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,                    # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,                    # image HSV-Value augmentation (fraction)
    'degrees': 0.0,                  # image rotation (+/- deg)
    'translate': 0.1,                # image translation (+/- fraction)
    'scale': 0.5,                    # image scale (+/- gain)
    'shear': 0.0,                    # image shear (+/- deg)
    'perspective': 0.0,              # image perspective (+/- fraction), range 0-0.001
    'flipud': 0.0,                   # image flip up-down (probability)
    'fliplr': 0.5,                   # image flip left-right (probability)
    'mosaic': 1.0,                   # image mosaic (probability)
    'mixup': 0.0,                    # image mixup (probability)

    # Anchor settings (optimized for pedestrians)
    'anchors': [
        [8, 18, 12, 28, 16, 38],       # P3/8 - small pedestrians
        [20, 48, 28, 68, 36, 88],      # P4/16 - medium pedestrians
        [48, 108, 64, 148, 80, 188],   # P5/32 - large pedestrians
    ]
}


# Model Configurations
MODEL_CONFIGS = {
    'hmal_tiny': {
        'backbone_channels': [32, 64, 128, 256],
        'fusion_channels': 128,
        'description': 'Lightweight HMAL for embedded devices',
        'target_fps': 60
    },
    'hmal_small': {
        'backbone_channels': [64, 128, 256, 512],
        'fusion_channels': 256,
        'description': 'Balanced HMAL for edge devices',
        'target_fps': 45
    },
    'hmal_medium': {
        'backbone_channels': [128, 256, 512, 1024],
        'fusion_channels': 512,
        'description': 'Standard HMAL configuration',
        'target_fps': 30
    },
    'hmal_large': {
        'backbone_channels': [256, 512, 1024, 2048],
        'fusion_channels': 1024,
        'description': 'High-capacity HMAL for maximum accuracy',
        'target_fps': 20
    }
}


# Dataset Configurations
DATASET_CONFIGS = {
    'LLVIP': {
        'num_classes': 1,
        'class_names': ['person'],
        'img_size': 640,
        'description': 'Low-Light Visible-Infrared Paired Dataset'
    },
    'FLIR': {
        'num_classes': 3,
        'class_names': ['person', 'car', 'bicycle'],
        'img_size': 640,
        'description': 'FLIR Thermal Dataset (Aligned)'
    },
    'M3FD': {
        'num_classes': 6,
        'class_names': ['person', 'car', 'bus', 'motorcycle', 'lamp', 'truck'],
        'img_size': 640,
        'description': 'Multi-Modal Multi-Spectral Fusion Dataset'
    },
    'KAIST': {
        'num_classes': 1,
        'class_names': ['person'],
        'img_size': 640,
        'description': 'KAIST Multispectral Pedestrian Dataset'
    }
}


# Training Configurations
TRAINING_CONFIGS = {
    'quick_test': {
        'epochs': 10,
        'batch_size': 8,
        'img_size': 320,
        'workers': 4,
        'description': 'Quick testing configuration'
    },
    'development': {
        'epochs': 50,
        'batch_size': 16,
        'img_size': 640,
        'workers': 8,
        'description': 'Development and debugging'
    },
    'production': {
        'epochs': 100,
        'batch_size': 32,
        'img_size': 640,
        'workers': 16,
        'description': 'Production training'
    },
    'high_quality': {
        'epochs': 200,
        'batch_size': 16,
        'img_size': 896,
        'workers': 12,
        'description': 'High-quality training for best results'
    }
}


class ConfigManager:
    """
    Manages project configurations and provides utilities for loading/saving configs.
    """

    def __init__(self, config_dir: str = './configs'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def save_hyperparameters(self, filepath: str, hyperparams: Dict[str, Any]):
        """Save hyperparameters to YAML file."""
        with open(filepath, 'w') as f:
            yaml.safe_dump(hyperparams, f, sort_keys=False)

    def load_hyperparameters(self, filepath: str) -> Dict[str, Any]:
        """Load hyperparameters from YAML file."""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    def create_dataset_config(self, name: str, train_rgb: str, train_ir: str,
                             val_rgb: str, val_ir: str, num_classes: int,
                             class_names: list) -> str:
        """
        Create dataset configuration file.

        Args:
            name: Dataset name
            train_rgb: Path to training RGB images
            train_ir: Path to training IR images
            val_rgb: Path to validation RGB images
            val_ir: Path to validation IR images
            num_classes: Number of classes
            class_names: List of class names

        Returns:
            Path to created config file
        """
        config = {
            'train_rgb': train_rgb,
            'train_ir': train_ir,
            'val_rgb': val_rgb,
            'val_ir': val_ir,
            'nc': num_classes,
            'names': class_names
        }

        config_path = self.config_dir / f'{name}.yaml'
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)

        return str(config_path)

    def get_model_config(self, model_size: str) -> Dict[str, Any]:
        """Get model configuration by size."""
        if model_size in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_size]
        else:
            raise ValueError(f"Unknown model size: {model_size}. "
                           f"Available: {list(MODEL_CONFIGS.keys())}")

    def get_training_config(self, config_name: str) -> Dict[str, Any]:
        """Get training configuration by name."""
        if config_name in TRAINING_CONFIGS:
            return TRAINING_CONFIGS[config_name]
        else:
            raise ValueError(f"Unknown training config: {config_name}. "
                           f"Available: {list(TRAINING_CONFIGS.keys())}")


def setup_project():
    """Initialize project structure and create default configurations."""

    # Create directory structure
    dirs = [
        'data/multispectral',
        'configs',
        'runs/train',
        'runs/eval',
        'weights',
        'logs',
        'docs'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Create default hyperparameters file
    config_manager = ConfigManager()
    config_manager.save_hyperparameters(
        'configs/hyp.default.yaml',
        DEFAULT_HYPERPARAMETERS
    )

    # Create model configs
    for model_name, model_cfg in MODEL_CONFIGS.items():
        config_manager.save_hyperparameters(
            f'configs/model_{model_name}.yaml',
            model_cfg
        )

    print("Project structure initialized successfully!")
    print(f"Created {len(dirs)} directories")
    print(f"Created {len(MODEL_CONFIGS) + 1} configuration files")


def print_config_info():
    """Print information about available configurations."""
    print("\n" + "="*80)
    print("VI PEDESTRIAN DETECTION SYSTEM - CONFIGURATION INFO")
    print("="*80)

    print("\nAvailable Model Sizes:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  - {name:15s}: {config['description']}")
        print(f"    {'':15s}  Target FPS: {config['target_fps']}")

    print("\nSupported Datasets:")
    for name, config in DATASET_CONFIGS.items():
        print(f"  - {name:15s}: {config['description']}")
        print(f"    {'':15s}  Classes: {config['num_classes']} {config['class_names']}")

    print("\nTraining Configurations:")
    for name, config in TRAINING_CONFIGS.items():
        print(f"  - {name:15s}: {config['description']}")
        print(f"    {'':15s}  Epochs: {config['epochs']}, Batch: {config['batch_size']}")

    print("\n" + "="*80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Project Configuration Manager')
    parser.add_argument('--setup', action='store_true', help='Initialize project structure')
    parser.add_argument('--info', action='store_true', help='Print configuration information')
    args = parser.parse_args()

    if args.setup:
        setup_project()

    if args.info or not any(vars(args).values()):
        print_config_info()
