"""
Evaluation Pipeline for RGB-IR Pedestrian Detection.
Comprehensive testing and analysis with multi-modal performance breakdown.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

from models.hmal import HMALDetector
from core.pedestrian_detector import PedestrianDetector
from core.data_preprocessing import create_pedestrian_dataloader
from core.evaluation_metrics import (
    PedestrianDetectionMetrics,
    ModalityContributionAnalyzer,
    RobustnessEvaluator
)
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """
    Orchestrates comprehensive evaluation of pedestrian detection models.
    Provides detailed analysis including per-scale metrics, modality contributions,
    and robustness assessment.
    """

    def __init__(self, config: Dict, device: torch.device):
        """
        Initialize evaluation orchestrator.

        Args:
            config: Configuration dictionary
            device: PyTorch device
        """
        self.config = config
        self.device = device

        # Setup output directories
        self.setup_output_dirs()

        # Load model
        self.model = self.load_model()

        # Initialize evaluators
        self.metrics_evaluator = PedestrianDetectionMetrics(
            num_classes=config['num_classes'],
            iou_threshold=config.get('iou_threshold', 0.5)
        )

        self.modality_analyzer = ModalityContributionAnalyzer()
        self.robustness_evaluator = RobustnessEvaluator()

        # Results storage
        self.all_predictions = []
        self.all_targets = []
        self.inference_times = []

    def setup_output_dirs(self):
        """Create output directories for results."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.detections_dir = self.output_dir / 'detections'
        self.visualizations_dir = self.output_dir / 'visualizations'
        self.metrics_dir = self.output_dir / 'metrics'

        for d in [self.detections_dir, self.visualizations_dir, self.metrics_dir]:
            d.mkdir(exist_ok=True)

    def load_model(self) -> nn.Module:
        """Load trained model from checkpoint."""
        checkpoint_path = self.config['checkpoint']

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Build model
        model_type = self.config.get('model_type', 'hmal')

        if model_type == 'hmal':
            model = HMALDetector(
                nc=self.config['num_classes'],
                anchors=self.config.get('anchors')
            )
        elif model_type == 'pedestrian':
            model = PedestrianDetector(
                num_classes=self.config['num_classes']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'ema_state_dict' in checkpoint:
            state_dict = checkpoint['ema_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Handle DataParallel wrapper
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        logger.info(f"Loaded model from: {checkpoint_path}")

        return model

    @torch.no_grad()
    def run_inference(self, rgb_image: torch.Tensor,
                     ir_image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run inference on a single image pair.

        Args:
            rgb_image: RGB image tensor [C, H, W]
            ir_image: Thermal image tensor [C, H, W]

        Returns:
            Tuple of (predictions, inference_time)
        """
        rgb_image = rgb_image.to(self.device)
        ir_image = ir_image.to(self.device)

        # Add batch dimension
        if rgb_image.dim() == 3:
            rgb_image = rgb_image.unsqueeze(0)
            ir_image = ir_image.unsqueeze(0)

        # Measure inference time
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()

        # Forward pass
        if hasattr(self.model, 'module'):
            predictions = self.model.module(rgb_image, ir_image)
        else:
            predictions = self.model(rgb_image, ir_image)

        end_time.record()
        torch.cuda.synchronize()

        inference_time = start_time.elapsed_time(end_time)

        return predictions, inference_time

    def postprocess_predictions(self, predictions: torch.Tensor,
                               conf_threshold: float = 0.25,
                               iou_threshold: float = 0.45) -> List[torch.Tensor]:
        """
        Postprocess raw predictions with NMS.

        Args:
            predictions: Raw model predictions
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of filtered predictions per image
        """
        # Apply NMS
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        processed = non_max_suppression(
            predictions,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            multi_label=False,
            agnostic=False
        )

        return processed

    def evaluate_dataset(self, dataloader) -> Dict[str, float]:
        """
        Evaluate model on entire dataset.

        Args:
            dataloader: Data loader for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")

        self.model.eval()
        self.metrics_evaluator.reset()

        for batch_idx, (images, targets, paths, shapes) in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device)

            # Split modalities
            rgb_images = images[:, :3, :, :].float() / 255.0
            ir_images = images[:, 3:, :, :].float() / 255.0

            # Run inference
            predictions, inference_time = self.run_inference(rgb_images, ir_images)
            self.inference_times.append(inference_time)

            # Postprocess predictions
            processed_preds = self.postprocess_predictions(
                predictions,
                conf_threshold=self.config.get('conf_threshold', 0.001),
                iou_threshold=self.config.get('iou_threshold', 0.6)
            )

            # Prepare targets per image
            batch_size = rgb_images.shape[0]
            targets_per_image = []

            for img_idx in range(batch_size):
                img_targets = targets[targets[:, 0] == img_idx, 1:]
                targets_per_image.append(img_targets)

            # Add to metrics evaluator
            self.metrics_evaluator.add_batch(processed_preds, targets_per_image)

            # Save detections if required
            if self.config.get('save_detections', False):
                self.save_detections(processed_preds, paths, batch_idx)

            # Visualize if required
            if self.config.get('save_visualizations', False) and batch_idx < 10:
                self.visualize_predictions(
                    rgb_images, ir_images,
                    processed_preds, targets_per_image,
                    paths, batch_idx
                )

        # Compute final metrics
        metrics = self.metrics_evaluator.evaluate()

        # Add timing statistics
        metrics['avg_inference_time_ms'] = np.mean(self.inference_times)
        metrics['fps'] = 1000.0 / metrics['avg_inference_time_ms']

        return metrics

    def save_detections(self, predictions: List[torch.Tensor],
                       paths: List[str], batch_idx: int):
        """Save detection results to file."""
        for img_idx, (pred, path) in enumerate(zip(predictions, paths)):
            if len(pred) == 0:
                continue

            # Convert to numpy
            pred_np = pred.cpu().numpy()

            # Create detection dict
            detections = []
            for det in pred_np:
                x1, y1, x2, y2, conf, cls = det[:6]
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': int(cls)
                })

            # Save to JSON
            detection_file = self.detections_dir / f"{Path(path).stem}.json"
            with open(detection_file, 'w') as f:
                json.dump(detections, f, indent=2)

    def visualize_predictions(self, rgb_images: torch.Tensor,
                             ir_images: torch.Tensor,
                             predictions: List[torch.Tensor],
                             targets: List[torch.Tensor],
                             paths: List[str], batch_idx: int):
        """Visualize predictions on images."""
        for img_idx in range(rgb_images.shape[0]):
            # Get images
            rgb_img = rgb_images[img_idx].cpu().permute(1, 2, 0).numpy()
            ir_img = ir_images[img_idx].cpu().permute(1, 2, 0).numpy()

            # Convert to uint8
            rgb_img = (rgb_img * 255).astype(np.uint8)
            ir_img = (ir_img * 255).astype(np.uint8)

            # Get predictions and targets
            pred = predictions[img_idx].cpu().numpy() if len(predictions[img_idx]) > 0 else np.array([])
            target = targets[img_idx].cpu().numpy() if len(targets[img_idx]) > 0 else np.array([])

            # Draw predictions (green)
            for det in pred:
                x1, y1, x2, y2, conf, cls = det[:6]
                cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(rgb_img, f"{conf:.2f}", (int(x1), int(y1)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw ground truth (red)
            if len(target) > 0:
                h, w = rgb_img.shape[:2]
                for gt in target:
                    if gt.shape[0] >= 5:
                        cls, xc, yc, box_w, box_h = gt[:5]
                        x1 = int((xc - box_w/2) * w)
                        y1 = int((yc - box_h/2) * h)
                        x2 = int((xc + box_w/2) * w)
                        y2 = int((yc + box_h/2) * h)
                        cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Create side-by-side visualization
            vis = np.hstack([rgb_img, ir_img])

            # Save visualization
            vis_file = self.visualizations_dir / f"batch_{batch_idx}_img_{img_idx}.jpg"
            cv2.imwrite(str(vis_file), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    def analyze_modality_contribution(self, dataloader) -> Dict[str, Dict]:
        """
        Analyze individual modality contributions.

        Args:
            dataloader: Data loader

        Returns:
            Dictionary with modality-specific performance
        """
        logger.info("Analyzing modality contributions...")

        # This would require separate RGB-only and IR-only models
        # For now, return placeholder
        return {
            'rgb_only': {'mAP': 0.0, 'recall': 0.0},
            'ir_only': {'mAP': 0.0, 'recall': 0.0},
            'fused': {'mAP': 0.0, 'recall': 0.0}
        }

    def assess_robustness(self, dataloader) -> Dict[str, Dict]:
        """
        Assess model robustness under different conditions.

        Args:
            dataloader: Data loader with condition labels

        Returns:
            Dictionary with condition-specific performance
        """
        logger.info("Assessing robustness...")

        # Evaluate robustness
        robustness_metrics = self.robustness_evaluator.evaluate_robustness()

        return robustness_metrics

    def generate_report(self, metrics: Dict[str, float]):
        """Generate comprehensive evaluation report."""
        report_file = self.metrics_dir / 'evaluation_report.txt'

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RGB-IR PEDESTRIAN DETECTION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("Overall Performance Metrics:\n")
            f.write("-" * 80 + "\n")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    f.write(f"{key:30s}: {value:.4f}\n")
                else:
                    f.write(f"{key:30s}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Evaluation report saved to: {report_file}")

        # Save metrics as JSON
        metrics_file = self.metrics_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        # Load test data
        test_loader = create_pedestrian_dataloader(
            rgb_root=self.config['test_rgb_path'],
            thermal_root=self.config['test_ir_path'],
            annotation_file=self.config['test_annotations'],
            batch_size=self.config['batch_size'],
            img_size=(self.config['img_size'], self.config['img_size']),
            augment=False,
            num_workers=self.config.get('num_workers', 4),
            shuffle=False
        )

        # Run evaluation
        metrics = self.evaluate_dataset(test_loader)

        # Generate report
        self.generate_report(metrics)

        # Print summary
        logger.info("\nEvaluation Summary:")
        logger.info(f"mAP@0.5: {metrics.get('mAP', 0):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"F1 Score: {metrics.get('f1', 0):.4f}")
        logger.info(f"LAMR: {metrics.get('LAMR', 1.0):.4f}")
        logger.info(f"Average Inference Time: {metrics.get('avg_inference_time_ms', 0):.2f} ms")
        logger.info(f"FPS: {metrics.get('fps', 0):.2f}")

        return metrics


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate RGB-IR Pedestrian Detector')

    # Data arguments
    parser.add_argument('--test-rgb', type=str, required=True, help='Path to test RGB images')
    parser.add_argument('--test-ir', type=str, required=True, help='Path to test IR images')
    parser.add_argument('--test-ann', type=str, required=True, help='Test annotations file')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='hmal', choices=['hmal', 'pedestrian'])
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')

    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--conf-threshold', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.6, help='IoU threshold for NMS')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./runs/eval', help='Output directory')
    parser.add_argument('--save-detections', action='store_true', help='Save detection results')
    parser.add_argument('--save-visualizations', action='store_true', help='Save visualizations')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Create configuration
    config = {
        'test_rgb_path': args.test_rgb,
        'test_ir_path': args.test_ir,
        'test_annotations': args.test_ann,
        'checkpoint': args.checkpoint,
        'model_type': args.model,
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'conf_threshold': args.conf_threshold,
        'iou_threshold': args.iou_threshold,
        'output_dir': args.output_dir,
        'save_detections': args.save_detections,
        'save_visualizations': args.save_visualizations,
        'num_workers': args.num_workers
    }

    # Setup device
    device = select_device(args.device)

    # Initialize evaluator
    evaluator = EvaluationOrchestrator(config, device)

    # Run evaluation
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
