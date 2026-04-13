"""
LC-Bayes R2 Manuscript Framework
================================
Core computational and data engine module for reproducible research.

Version: 2.0.0
Author: LC-Bayes R2 Consortium
"""

__version__ = "2.0.0"
__author__ = "LC-Bayes R2 Consortium"

import logging

from .data_engine import LCBayesDataSynthesizer, SynthesizedData
from .models import (
    SelectionAwareLOEUF,
    FetalMaternalHMM,
    BayesianVarianceComponent,
    SSgBLUPWeights
)

from .pipeline_wrappers import AlphaSimRRunner, RelateCluesRunner
from .pipeline_wrappers import fallback_to_mock_if_missing

try:
    from .visualizer import PaperVisualizer
except ImportError:
    PaperVisualizer = None
    logging.warning("Visualization dependencies (matplotlib/seaborn) not available. Visualizer disabled.")

__all__ = [
    "LCBayesDataSynthesizer",
    "SynthesizedData",
    "SelectionAwareLOEUF",
    "FetalMaternalHMM",
    "BayesianVarianceComponent",
    "SSgBLUPWeights",
    "AlphaSimRRunner",
    "RelateCluesRunner",
    "fallback_to_mock_if_missing",
    "PaperVisualizer"
]
