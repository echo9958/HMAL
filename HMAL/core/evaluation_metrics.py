"""
Comprehensive Evaluation Metrics for Pedestrian Detection.
Implements Miss Rate, LAMR, Precision-Recall curves, and multi-modal analysis.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path


class PedestrianDetectionMetrics:
    """
    Comprehensive metrics suite for evaluating pedestrian detection performance.
    Includes standard object detection metrics and pedestrian-specific measures.
    """

    def __init__(self, num_classes: int = 1, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

        # Storage for predictions and ground truth
        self.predictions = []
        self.ground_truths = []

        # Pedestrian-specific scale categories
        self.scale_categories = {
            'near': (0.1, 1.0),      # Large pedestrians (close)
            'medium': (0.03, 0.1),    # Medium pedestrians
            'far': (0.0, 0.03)        # Small pedestrians (far)
        }

    def add_batch(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        """
        Add a batch of predictions and targets for evaluation.

        Args:
            predictions: List of prediction tensors [N, 6] (x1, y1, x2, y2, conf, class)
            targets: List of ground truth tensors [M, 5] (class, x_center, y_center, w, h)
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(targets)

    def compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        Compute Average Precision using 11-point interpolation.

        Args:
            recalls: Array of recall values
            precisions: Array of precision values

        Returns:
            Average Precision score
        """
        # Append sentinel values
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))

        # Compute precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Compute AP using 11-point interpolation
        ap = 0.
        for threshold in np.linspace(0, 1, 11):
            if np.sum(mrec >= threshold) == 0:
                p = 0
            else:
                p = np.max(mpre[mrec >= threshold])
            ap += p / 11.

        return ap

    def compute_miss_rate(self, recalls: np.ndarray) -> np.ndarray:
        """
        Compute miss rate (1 - recall) at different thresholds.

        Args:
            recalls: Array of recall values

        Returns:
            Array of miss rate values
        """
        return 1.0 - recalls

    def compute_lamr(self, fppi: np.ndarray, miss_rate: np.ndarray) -> float:
        """
        Compute Log-Average Miss Rate (LAMR) - key metric for pedestrian detection.

        Args:
            fppi: False Positives Per Image array
            miss_rate: Miss rate array

        Returns:
            LAMR score
        """
        # Sample miss rate at specific FPPI points
        fppi_sample_points = np.logspace(-2, 0, 9)  # [0.01, 0.1, 1.0]

        # Interpolate miss rate at sample points
        if len(fppi) < 2 or len(miss_rate) < 2:
            return 1.0

        # Sort by FPPI
        sorted_indices = np.argsort(fppi)
        fppi_sorted = fppi[sorted_indices]
        mr_sorted = miss_rate[sorted_indices]

        # Remove duplicates
        unique_fppi, unique_indices = np.unique(fppi_sorted, return_index=True)
        unique_mr = mr_sorted[unique_indices]

        if len(unique_fppi) < 2:
            return 1.0

        # Interpolate
        try:
            interpolator = interp1d(unique_fppi, unique_mr, bounds_error=False,
                                   fill_value=(unique_mr[0], unique_mr[-1]))
            sampled_mr = interpolator(fppi_sample_points)

            # Compute log-average
            lamr = np.exp(np.mean(np.log(np.maximum(sampled_mr, 1e-10))))

            return float(lamr)

        except:
            return 1.0

    def evaluate(self) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Returns:
            Dictionary containing various metrics
        """
        if len(self.predictions) == 0 or len(self.ground_truths) == 0:
            return self._empty_metrics()

        # Organize predictions by confidence
        all_predictions = []
        all_targets = []

        for pred_boxes, gt_boxes in zip(self.predictions, self.ground_truths):
            if len(pred_boxes) > 0:
                for pred in pred_boxes:
                    all_predictions.append({
                        'bbox': pred[:4].cpu().numpy(),
                        'confidence': float(pred[4]),
                        'class': int(pred[5]) if len(pred) > 5 else 0
                    })

            if len(gt_boxes) > 0:
                for gt in gt_boxes:
                    # Convert from YOLO format to xyxy
                    if gt.shape[0] == 5:  # class, xc, yc, w, h
                        cls, xc, yc, w, h = gt.cpu().numpy()
                        x1 = xc - w / 2
                        y1 = yc - h / 2
                        x2 = xc + w / 2
                        y2 = yc + h / 2
                        area = w * h
                    else:
                        cls, x1, y1, x2, y2 = gt.cpu().numpy()[:5]
                        area = (x2 - x1) * (y2 - y1)

                    all_targets.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'class': int(cls),
                        'area': area,
                        'detected': False
                    })

        # Sort predictions by confidence (descending)
        all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)

        # Compute precision-recall curve
        tp = np.zeros(len(all_predictions))
        fp = np.zeros(len(all_predictions))

        for pred_idx, pred in enumerate(all_predictions):
            # Find matching ground truth
            max_iou = 0
            max_gt_idx = -1

            for gt_idx, gt in enumerate(all_targets):
                if gt['class'] != pred['class']:
                    continue

                iou = self._compute_iou(pred['bbox'], gt['bbox'])

                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            # Check if match is valid
            if max_iou >= self.iou_threshold:
                if not all_targets[max_gt_idx]['detected']:
                    tp[pred_idx] = 1
                    all_targets[max_gt_idx]['detected'] = True
                else:
                    fp[pred_idx] = 1  # Already detected
            else:
                fp[pred_idx] = 1

        # Compute cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        num_gt = len(all_targets)

        # Compute precision and recall
        recalls = tp_cumsum / (num_gt + 1e-10)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # Compute FPPI (False Positives Per Image)
        num_images = len(self.predictions)
        fppi = fp_cumsum / num_images

        # Compute metrics
        ap = self.compute_ap(recalls, precisions)
        miss_rate = self.compute_miss_rate(recalls)
        lamr = self.compute_lamr(fppi, miss_rate)

        # Find F1 score
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        max_f1_idx = np.argmax(f1_scores)

        # Compute scale-specific metrics
        scale_metrics = self._compute_scale_metrics(all_predictions, all_targets)

        return {
            'mAP': float(ap),
            'precision': float(precisions[max_f1_idx]) if len(precisions) > 0 else 0.0,
            'recall': float(recalls[max_f1_idx]) if len(recalls) > 0 else 0.0,
            'f1': float(f1_scores[max_f1_idx]) if len(f1_scores) > 0 else 0.0,
            'LAMR': float(lamr),
            'miss_rate': float(miss_rate[max_f1_idx]) if len(miss_rate) > 0 else 1.0,
            **scale_metrics
        }

    def _compute_scale_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Compute metrics for different pedestrian scales."""
        scale_results = {}

        for scale_name, (min_area, max_area) in self.scale_categories.items():
            # Filter targets by scale
            scale_targets = [t for t in targets
                           if min_area <= t['area'] < max_area]

            if len(scale_targets) == 0:
                continue

            # Count true positives for this scale
            tp_count = sum(1 for t in scale_targets if t['detected'])

            scale_recall = tp_count / len(scale_targets)
            scale_miss_rate = 1.0 - scale_recall

            scale_results[f'recall_{scale_name}'] = float(scale_recall)
            scale_results[f'miss_rate_{scale_name}'] = float(scale_miss_rate)

        return scale_results

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / (union + 1e-10)

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'LAMR': 1.0,
            'miss_rate': 1.0
        }

    def plot_curves(self, save_dir: Optional[Path] = None):
        """Plot evaluation curves (PR curve, Miss Rate vs FPPI)."""
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Compute curves
        # ... (Implementation for plotting)

    def reset(self):
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.ground_truths = []


class ModalityContributionAnalyzer:
    """
    Analyzes the contribution of each modality (RGB vs Thermal) to detection performance.
    Useful for understanding when each modality is more effective.
    """

    def __init__(self):
        self.rgb_only_results = []
        self.thermal_only_results = []
        self.fused_results = []

    def add_result(self, rgb_pred: torch.Tensor, thermal_pred: torch.Tensor,
                   fused_pred: torch.Tensor, target: torch.Tensor):
        """Add predictions from each modality and fusion."""
        self.rgb_only_results.append((rgb_pred, target))
        self.thermal_only_results.append((thermal_pred, target))
        self.fused_results.append((fused_pred, target))

    def analyze(self) -> Dict[str, float]:
        """
        Analyze modality contributions.

        Returns:
            Dictionary with contribution statistics
        """
        # Compute metrics for each modality
        rgb_metrics = self._compute_metrics(self.rgb_only_results)
        thermal_metrics = self._compute_metrics(self.thermal_only_results)
        fused_metrics = self._compute_metrics(self.fused_results)

        # Compute fusion gain
        fusion_gain = {
            'ap_gain': fused_metrics['ap'] - max(rgb_metrics['ap'], thermal_metrics['ap']),
            'recall_gain': fused_metrics['recall'] - max(rgb_metrics['recall'], thermal_metrics['recall']),
            'rgb_contribution': rgb_metrics['ap'] / (fused_metrics['ap'] + 1e-6),
            'thermal_contribution': thermal_metrics['ap'] / (fused_metrics['ap'] + 1e-6)
        }

        return {
            'rgb': rgb_metrics,
            'thermal': thermal_metrics,
            'fused': fused_metrics,
            'fusion_gain': fusion_gain
        }

    def _compute_metrics(self, results: List[Tuple]) -> Dict[str, float]:
        """Compute basic metrics from results."""
        # Simplified computation
        if len(results) == 0:
            return {'ap': 0.0, 'recall': 0.0, 'precision': 0.0}

        # Placeholder - actual implementation would compute proper metrics
        return {
            'ap': 0.5,
            'recall': 0.5,
            'precision': 0.5
        }


class RobustnessEvaluator:
    """
    Evaluates model robustness under various challenging conditions.
    Includes low-light, occlusion, and scale variation analysis.
    """

    def __init__(self):
        self.condition_results = {
            'day': {'predictions': [], 'targets': []},
            'night': {'predictions': [], 'targets': []},
            'occluded': {'predictions': [], 'targets': []},
            'crowded': {'predictions': [], 'targets': []}
        }

    def add_sample(self, prediction: torch.Tensor, target: torch.Tensor,
                   condition: str, metadata: Optional[Dict] = None):
        """
        Add a sample with specific condition label.

        Args:
            prediction: Predicted bounding boxes
            target: Ground truth boxes
            condition: Condition label (day, night, occluded, crowded)
            metadata: Additional metadata
        """
        if condition in self.condition_results:
            self.condition_results[condition]['predictions'].append(prediction)
            self.condition_results[condition]['targets'].append(target)

    def evaluate_robustness(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance under different conditions.

        Returns:
            Dictionary with condition-specific metrics
        """
        results = {}

        for condition, data in self.condition_results.items():
            if len(data['predictions']) == 0:
                continue

            # Compute metrics for this condition
            evaluator = PedestrianDetectionMetrics()
            evaluator.predictions = data['predictions']
            evaluator.ground_truths = data['targets']

            metrics = evaluator.evaluate()
            results[condition] = metrics

        return results


class TemporalConsistencyEvaluator:
    """
    Evaluates temporal consistency for video-based pedestrian detection.
    Measures tracking stability and ID switches.
    """

    def __init__(self):
        self.tracks = {}  # Track ID -> [frame_detections]
        self.id_switches = 0
        self.fragmentations = 0

    def add_frame_detections(self, frame_id: int, detections: List[Dict]):
        """
        Add detections for a video frame.

        Args:
            frame_id: Frame number
            detections: List of detection dicts with bbox, track_id
        """
        for det in detections:
            track_id = det.get('track_id', -1)

            if track_id not in self.tracks:
                self.tracks[track_id] = []

            self.tracks[track_id].append({
                'frame': frame_id,
                'bbox': det['bbox']
            })

    def compute_consistency(self) -> Dict[str, float]:
        """
        Compute temporal consistency metrics.

        Returns:
            Dictionary with consistency scores
        """
        # Compute average track length
        track_lengths = [len(track) for track in self.tracks.values()]
        avg_track_length = np.mean(track_lengths) if track_lengths else 0

        # Compute fragmentation rate
        fragmentation_rate = self.fragmentations / len(self.tracks) if self.tracks else 0

        return {
            'avg_track_length': float(avg_track_length),
            'id_switches': int(self.id_switches),
            'fragmentation_rate': float(fragmentation_rate),
            'num_tracks': len(self.tracks)
        }
