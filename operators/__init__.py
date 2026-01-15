"""W&B Plugin Operators.

This package contains all the operators for the W&B plugin.
"""
import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

from .log_wandb_run import LogWandBRun
from .open_wandb_panel import OpenWandBPanel
from .show_wandb_run import ShowWandBRun
from .get_wandb_run_info import GetWandBRunInfo
from .show_wandb_report import ShowWandBReport
from .log_fiftyone_view_to_wandb import LogFiftyOneViewToWandB
from .log_model_predictions import LogModelPredictions
from .load_view_from_wandb import LoadViewFromWandB
from .train_yolo_model import TrainYOLOModel
from .apply_yolo_model_from_registry import ApplyYOLOModelFromRegistry

__all__ = [
    "LogWandBRun",
    "OpenWandBPanel",
    "ShowWandBRun",
    "GetWandBRunInfo",
    "ShowWandBReport",
    "LogFiftyOneViewToWandB",
    "LogModelPredictions",
    "LoadViewFromWandB",
    "TrainYOLOModel",
    "ApplyYOLOModelFromRegistry",
]

