# HMAL: Hierarchical Modality Advantage Learning for RGB-IR Pedestrian Detection

Advanced RGB-Infrared pedestrian detection framework implementing hierarchical modality advantage learning with multi-scale fusion strategies. This system achieves state-of-the-art performance on challenging pedestrian detection benchmarks through adaptive modality weighting and cross-level feature integration.

### Evaluation Metrics
- **Standard Metrics**: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
- **Pedestrian-Specific**: Log-Average Miss Rate (LAMR), Miss Rate vs FPPI curves
- **Scale-Specific Analysis**: Performance breakdown by pedestrian distance (near/medium/far)
- **Modality Contribution**: Individual RGB/thermal performance and fusion gain analysis
- **Robustness Assessment**: Day/night, occlusion, and crowded scene evaluation

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

