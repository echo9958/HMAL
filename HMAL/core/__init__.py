"""
Core modules for RGB-Infrared Pedestrian Detection System.
Implements hierarchical modality advantage learning framework.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .pedestrian_detector import PedestrianDetector
from .fusion_strategies import (
    AdaptiveFusionModule,
    HierarchicalFusion,
    ModalityRouter
)
from .feature_extractor import MultiModalFeatureExtractor

__all__ = [
    'PedestrianDetector',
    'AdaptiveFusionModule',
    'HierarchicalFusion',
    'ModalityRouter',
    'MultiModalFeatureExtractor'
]
