"""
Visualization Tools for RGB-IR Pedestrian Detection.
Provides utilities for visualizing detections, features, and analysis results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch


class DetectionVisualizer:
    """
    Visualizes detection results on RGB and thermal images.
    Supports side-by-side comparison and confidence visualization.
    """

    def __init__(self, class_names: List[str] = None, colormap: str = 'tab10'):
        """
        Args:
            class_names: List of class names
            colormap: Matplotlib colormap for class colors
        """
        self.class_names = class_names or ['pedestrian']
        self.num_classes = len(self.class_names)

        # Generate colors for each class
        cmap = plt.get_cmap(colormap)
        self.colors = [
            tuple(int(c * 255) for c in cmap(i / self.num_classes)[:3])
            for i in range(self.num_classes)
        ]

    def draw_boxes(self, image: np.ndarray, boxes: np.ndarray,
                   scores: Optional[np.ndarray] = None,
                   classes: Optional[np.ndarray] = None,
                   line_thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: Input image [H, W, 3]
            boxes: Bounding boxes [N, 4] in xyxy format
            scores: Confidence scores [N]
            classes: Class indices [N]
            line_thickness: Thickness of bounding box lines

        Returns:
            Image with drawn boxes
        """
        image = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])

            # Get class and color
            cls_idx = int(classes[i]) if classes is not None else 0
            color = self.colors[cls_idx % len(self.colors)]

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

            # Draw label
            label = self.class_names[cls_idx]
            if scores is not None:
                label += f' {scores[i]:.2f}'

            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                image, label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

        return image

    def visualize_dual_modality(self, rgb_image: np.ndarray, thermal_image: np.ndarray,
                                predictions: np.ndarray,
                                ground_truth: Optional[np.ndarray] = None,
                                save_path: Optional[Path] = None) -> np.ndarray:
        """
        Create side-by-side visualization of RGB and thermal detections.

        Args:
            rgb_image: RGB image
            thermal_image: Thermal image
            predictions: Predicted boxes [N, 6] (x1, y1, x2, y2, conf, class)
            ground_truth: Ground truth boxes [M, 5] (class, xc, yc, w, h)
            save_path: Path to save visualization

        Returns:
            Combined visualization image
        """
        # Draw predictions in green
        if len(predictions) > 0:
            pred_boxes = predictions[:, :4]
            pred_scores = predictions[:, 4]
            pred_classes = predictions[:, 5]

            rgb_with_pred = self.draw_boxes(rgb_image, pred_boxes, pred_scores, pred_classes)
            thermal_with_pred = self.draw_boxes(thermal_image, pred_boxes, pred_scores, pred_classes)
        else:
            rgb_with_pred = rgb_image.copy()
            thermal_with_pred = thermal_image.copy()

        # Draw ground truth in red if provided
        if ground_truth is not None and len(ground_truth) > 0:
            h, w = rgb_image.shape[:2]

            for gt in ground_truth:
                if len(gt) >= 5:
                    cls, xc, yc, box_w, box_h = gt[:5]

                    # Convert from YOLO format to xyxy
                    x1 = int((xc - box_w/2) * w)
                    y1 = int((yc - box_h/2) * h)
                    x2 = int((xc + box_w/2) * w)
                    y2 = int((yc + box_h/2) * h)

                    # Draw on both images
                    cv2.rectangle(rgb_with_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(thermal_with_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Add legend
        legend_h = 60
        legend = np.ones((legend_h, w * 2, 3), dtype=np.uint8) * 255

        cv2.putText(legend, "Green: Predictions", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(legend, "Red: Ground Truth", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(legend, "RGB", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(legend, f"Thermal ({len(predictions)} detections)",
                   (w//2 - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Combine images
        combined = np.hstack([rgb_with_pred, thermal_with_pred])
        final = np.vstack([legend, combined])

        if save_path is not None:
            cv2.imwrite(str(save_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

        return final


class FeatureMapVisualizer:
    """
    Visualizes intermediate feature maps from the detection model.
    Useful for understanding model behavior and debugging.
    """

    def __init__(self):
        pass

    def visualize_feature_map(self, feature_map: torch.Tensor,
                             num_features: int = 16,
                             save_path: Optional[Path] = None) -> np.ndarray:
        """
        Visualize feature map channels.

        Args:
            feature_map: Feature map tensor [C, H, W]
            num_features: Number of feature channels to visualize
            save_path: Path to save visualization

        Returns:
            Visualization image
        """
        if feature_map.dim() == 4:
            feature_map = feature_map[0]  # Take first in batch

        feature_map = feature_map.cpu().detach().numpy()
        num_channels = min(num_features, feature_map.shape[0])

        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_channels)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(num_channels):
            feat = feature_map[i]

            # Normalize to [0, 1]
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

            axes[i].imshow(feat, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Ch {i}', fontsize=8)

        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return fig

    def visualize_attention_maps(self, attention_weights: torch.Tensor,
                                 image: np.ndarray,
                                 save_path: Optional[Path] = None) -> np.ndarray:
        """
        Overlay attention maps on original image.

        Args:
            attention_weights: Attention weight tensor [H, W]
            image: Original image
            save_path: Path to save visualization

        Returns:
            Visualization with attention overlay
        """
        # Normalize attention weights
        attention = attention_weights.cpu().detach().numpy()
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        # Resize to match image size
        h, w = image.shape[:2]
        attention_resized = cv2.resize(attention, (w, h))

        # Apply colormap
        attention_colored = plt.cm.jet(attention_resized)[:, :, :3]
        attention_colored = (attention_colored * 255).astype(np.uint8)

        # Blend with original image
        alpha = 0.5
        overlay = cv2.addWeighted(image, 1-alpha, attention_colored, alpha, 0)

        if save_path is not None:
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return overlay


class MetricsVisualizer:
    """
    Visualizes evaluation metrics and performance curves.
    """

    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_pr_curve(self, precisions: np.ndarray, recalls: np.ndarray,
                      ap_score: float, save_path: Optional[Path] = None):
        """
        Plot Precision-Recall curve.

        Args:
            precisions: Precision values
            recalls: Recall values
            ap_score: Average Precision score
            save_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))

        plt.plot(recalls, precisions, linewidth=2, label=f'AP = {ap_score:.3f}')
        plt.fill_between(recalls, precisions, alpha=0.2)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_miss_rate_curve(self, fppi: np.ndarray, miss_rates: np.ndarray,
                            lamr_score: float, save_path: Optional[Path] = None):
        """
        Plot Miss Rate vs FPPI curve (log-log scale).

        Args:
            fppi: False Positives Per Image
            miss_rates: Miss rate values
            lamr_score: Log-Average Miss Rate
            save_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))

        plt.loglog(fppi, miss_rates, linewidth=2, label=f'LAMR = {lamr_score:.3f}')
        plt.fill_between(fppi, miss_rates, alpha=0.2)

        plt.xlabel('False Positives Per Image (FPPI)', fontsize=12)
        plt.ylabel('Miss Rate', fontsize=12)
        plt.title('Miss Rate vs FPPI', fontsize=14, fontweight='bold')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(fontsize=11)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_scale_performance(self, scale_metrics: Dict[str, float],
                              save_path: Optional[Path] = None):
        """
        Plot performance across different pedestrian scales.

        Args:
            scale_metrics: Dictionary with scale-specific metrics
            save_path: Path to save plot
        """
        scales = list(scale_metrics.keys())
        values = list(scale_metrics.values())

        plt.figure(figsize=(10, 6))

        bars = plt.bar(scales, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

        plt.xlabel('Pedestrian Scale', fontsize=12)
        plt.ylabel('Recall', fontsize=12)
        plt.title('Performance by Pedestrian Scale', fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        plt.grid(True, axis='y', alpha=0.3)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                             class_names: List[str],
                             save_path: Optional[Path] = None):
        """
        Plot confusion matrix heatmap.

        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))

        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion'})

        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class TrainingVisualizer:
    """
    Visualizes training progress and metrics over epochs.
    """

    def __init__(self):
        pass

    def plot_training_curves(self, train_losses: List[float],
                            val_losses: List[float],
                            val_metrics: Dict[str, List[float]],
                            save_path: Optional[Path] = None):
        """
        Plot training and validation curves.

        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            val_metrics: Dictionary of validation metrics per epoch
            save_path: Path to save plot
        """
        epochs = list(range(1, len(train_losses) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curves
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # mAP curve
        if 'mAP' in val_metrics:
            axes[0, 1].plot(epochs, val_metrics['mAP'], linewidth=2, color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].set_title('Validation mAP')
            axes[0, 1].grid(True, alpha=0.3)

        # Precision and Recall
        if 'precision' in val_metrics and 'recall' in val_metrics:
            axes[1, 0].plot(epochs, val_metrics['precision'], label='Precision', linewidth=2)
            axes[1, 0].plot(epochs, val_metrics['recall'], label='Recall', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Precision and Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # F1 Score
        if 'f1' in val_metrics:
            axes[1, 1].plot(epochs, val_metrics['f1'], linewidth=2, color='purple')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('Validation F1 Score')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
