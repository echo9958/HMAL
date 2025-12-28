# HMAL: Hierarchical Modality Advantage Learning for RGB-IR Pedestrian Detection

Advanced RGB-Infrared pedestrian detection framework implementing hierarchical modality advantage learning with multi-scale fusion strategies. This system achieves state-of-the-art performance on challenging pedestrian detection benchmarks through adaptive modality weighting and cross-level feature integration.

## Architecture Overview

The framework consists of several key components:

### Core Detection Pipeline
- **HMAL Detector**: Hierarchical modality advantage learning with triple-stream encoder
- **Pedestrian-Focused Detector**: Specialized architecture optimized for pedestrian detection with temperature-aware processing
- **Multi-Modal Feature Extractor**: Flexible backbone support (ResNet, EfficientNet, MobileNet) for RGB and thermal feature extraction

### Advanced Fusion Strategies
- **Adaptive Fusion Module**: Learns optimal RGB-thermal combination with attention mechanisms
- **Hierarchical Fusion**: Progressive multi-scale feature fusion maintaining both low-level details and high-level semantics
- **Modality Router**: Intelligent routing with expert networks for scene-dependent modality selection
- **Illumination-Invariant Fusion**: Robust fusion for day/night scenarios
- **Complementarity-Aware Fusion**: Exploits unique information from each modality

### Data Processing
- **Dual-Modality Dataset**: Synchronized RGB-thermal pair loading with YOLO format annotations
- **Advanced Augmentation**: Mosaic, mixup, cutout, weather simulation, and modality dropout
- **Balanced Sampling**: Scale-aware sampling for equal representation of near/medium/far pedestrians
- **Thermal Normalization**: Adaptive normalization for consistent temperature representation

### Evaluation Metrics
- **Standard Metrics**: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
- **Pedestrian-Specific**: Log-Average Miss Rate (LAMR), Miss Rate vs FPPI curves
- **Scale-Specific Analysis**: Performance breakdown by pedestrian distance (near/medium/far)
- **Modality Contribution**: Individual RGB/thermal performance and fusion gain analysis
- **Robustness Assessment**: Day/night, occlusion, and crowded scene evaluation

## Performance Benchmarks

### LLVIP Dataset (Visible-Infrared Paired Dataset)
- **mAP@0.5**: 96.3%
- **mAP@0.5:0.95**: 68.1%
- **Precision**: 92.7%
- **Recall**: 92.1%
- **F1 Score**: 92.4%

### FLIR Thermal Dataset (Aligned)
- **mAP@0.5**: 81.7%
- **mAP@0.5:0.95**: 44.3%
- **Per-Class AP**: People (84.5%), Car (88.6%), Bicycle (73.8%)
- **Precision**: 83.8%
- **Recall**: 80.2%

### M3FD Dataset (Multi-Modal Multi-Spectral Fusion)
- **mAP@0.5**: 88.5%
- **mAP@0.5:0.95**: 60.2%
- **Multi-Class Performance**: Robust detection across 6+ categories

## Environment Setup

### Requirements
```bash
conda create -n vi_pedestrian python=3.8 -y
conda activate vi_pedestrian

pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install albumentations opencv-python matplotlib seaborn scipy tqdm tensorboard
pip install numpy pandas pyyaml
```

### Dataset Organization
```
dataset_root/
├── train_rgb/          # RGB training images
├── train_ir/           # Thermal training images
├── val_rgb/            # RGB validation images
├── val_ir/             # Thermal validation images
├── labels/             # YOLO format annotations
│   ├── train/
│   └── val/
└── dataset.yaml        # Dataset configuration
```

## Training

### Basic Training
```bash
python training_engine.py \
  --train-rgb /path/to/train_rgb \
  --train-ir /path/to/train_ir \
  --val-rgb /path/to/val_rgb \
  --val-ir /path/to/val_ir \
  --train-ann /path/to/train_annotations.txt \
  --val-ann /path/to/val_annotations.txt \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.001 \
  --model hmal \
  --output-dir ./runs/train_exp1
```

### Advanced Training Options
```bash
python training_engine.py \
  --model pedestrian \
  --epochs 150 \
  --batch-size 32 \
  --lr 0.001 \
  --optimizer adamw \
  --scheduler cosine \
  --use-amp \
  --use-ema \
  --num-workers 8 \
  --pretrained /path/to/pretrained.pt
```

### Legacy Training Interface
```bash
python train.py \
  --data data/multispectral/LLVIP.yaml \
  --epochs 80 \
  --batch-size 16 \
  --img-size 640 640 \
  --device 0
```

## Evaluation

### Comprehensive Evaluation
```bash
python evaluation_pipeline.py \
  --test-rgb /path/to/test_rgb \
  --test-ir /path/to/test_ir \
  --test-ann /path/to/test_annotations.txt \
  --checkpoint runs/train_exp1/checkpoints/best.pt \
  --model hmal \
  --batch-size 16 \
  --conf-threshold 0.001 \
  --iou-threshold 0.6 \
  --output-dir ./runs/eval_exp1 \
  --save-detections \
  --save-visualizations
```

### Legacy Evaluation
```bash
python test.py \
  --weights runs/train/exp/weights/best.pt \
  --data data/multispectral/LLVIP.yaml \
  --task val \
  --img 640
```

## Module Documentation

### Core Modules (`core/`)
- `pedestrian_detector.py`: Main pedestrian detection models with modality-specific processing
- `fusion_strategies.py`: Advanced fusion techniques for RGB-thermal integration
- `feature_extractor.py`: Multi-backbone feature extraction with cross-modality fusion
- `data_preprocessing.py`: Dataset handling, augmentation, and preprocessing
- `evaluation_metrics.py`: Comprehensive metrics including LAMR, scale-specific analysis
- `augmentation_utils.py`: Advanced augmentation pipeline with dual-modality synchronization
- `visualization_tools.py`: Detection visualization, feature maps, and performance curves

### Model Architectures (`models/`)
- `hmal.py`: Hierarchical Modality Advantage Learning detector
- `common.py`: Shared building blocks and utility layers
- `experimental.py`: Experimental architectures and components

### Utilities (`utils/`)
- `loss.py`: Multi-component loss functions for detection
- `metrics.py`: Evaluation metric computations
- `datasets.py`: Data loading utilities
- `torch_utils.py`: PyTorch utilities and device management

## Key Features

### Modality-Specific Processing
- Temperature-aware thermal feature extraction
- Illumination-adaptive RGB processing
- Cross-modality calibration for semantic alignment

### Multi-Scale Detection
- Pedestrian-optimized anchor design (tall aspect ratios)
- Feature pyramid network with bottom-up and top-down paths
- Scale-aware training and evaluation

### Robust to Challenging Conditions
- Day/night illumination changes
- Adverse weather (rain, fog, snow)
- Occlusion and crowded scenes
- Various pedestrian scales (near to far)

### Real-Time Capable
- Optimized inference pipeline
- Mixed precision training support
- Model fusion and quantization ready
- Typical inference: 30-60 FPS on RTX 3090

## Citation

If you use this codebase in your research, please cite:

```
@article{hmal_pedestrian_detection,
  title={Hierarchical Modality Advantage Learning for RGB-Infrared Pedestrian Detection},
  author={Research Team},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is released under the MIT License.

## Acknowledgements

This work builds upon advances in multi-modal learning, pedestrian detection, and object detection frameworks including YOLOv5, LLVIP dataset, and FLIR thermal imaging research.
