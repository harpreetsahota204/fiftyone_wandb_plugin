"""Weights & Biases Experiment Tracking plugin.

Copyright 2017-2024, Voxel51, Inc.
`voxel51.com <https://voxel51.com/>`_
"""

import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

from .operators import (
    LogWandBRun,
    OpenWandBPanel,
    ShowWandBRun,
    GetWandBRunInfo,
    ShowWandBReport,
    LogFiftyOneViewToWandB,
)


def register(plugin):
    """Register all W&B operators with the plugin system."""
    plugin.register(OpenWandBPanel)
    plugin.register(GetWandBRunInfo)
    plugin.register(LogWandBRun)
    plugin.register(ShowWandBRun)
    plugin.register(ShowWandBReport)
    plugin.register(LogFiftyOneViewToWandB)
