"""Helper functions for W&B plugin operators.

This module contains all shared utility functions used by the W&B operators.
"""

import json
import os
import tempfile
from datetime import datetime
from bson import json_util

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

DEFAULT_WANDB_URL = "https://wandb.ai"


# ===========================
# W&B Configuration Helpers
# ===========================

def get_wandb_config(ctx):
    """Get W&B configuration from secrets with fallbacks"""
    config = {
        "api_key": ctx.secret("FIFTYONE_WANDB_API_KEY"),
        "entity": ctx.secret("FIFTYONE_WANDB_ENTITY"),
        "project": ctx.secret("FIFTYONE_WANDB_PROJECT"),
    }
    return config


def ensure_wandb_login(ctx):
    """Ensure W&B is logged in"""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Install it with: pip install wandb")
    
    api_key = ctx.secret("FIFTYONE_WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    # If no API key, wandb will use cached login or prompt


def get_wandb_api(ctx):
    """Initialize W&B API client"""
    ensure_wandb_login(ctx)
    api = wandb.Api()
    return api


# ===========================
# W&B Project/Run Helpers
# ===========================

def get_project_path(ctx, project_name):
    """Get W&B project path in entity/project format"""
    config = get_wandb_config(ctx)
    entity = config["entity"]
    if not entity:
        # Try to get from API
        api = get_wandb_api(ctx)
        entity = api.default_entity
    return f"{entity}/{project_name}"


def get_wandb_run(ctx, project_name, run_id=None, run_name=None):
    """Get W&B run by ID or name"""
    api = get_wandb_api(ctx)
    project_path = get_project_path(ctx, project_name)
    
    if run_id:
        # Get by ID
        return api.run(f"{project_path}/{run_id}")
    elif run_name:
        # Search by name
        runs = api.runs(project_path, filters={"display_name": run_name})
        if runs:
            return runs[0]
    
    return None


def get_project_url(ctx, project_name):
    """Construct W&B project URL"""
    config = get_wandb_config(ctx)
    entity = config["entity"]
    if not entity:
        api = get_wandb_api(ctx)
        entity = api.default_entity
    return f"{DEFAULT_WANDB_URL}/{entity}/{project_name}"


def get_run_url(ctx, project_name, run_id):
    """Construct W&B run URL"""
    project_url = get_project_url(ctx, project_name)
    return f"{project_url}/runs/{run_id}"


def format_run_name(run_name):
    """Format run name for FiftyOne (replace hyphens with underscores)"""
    return run_name.replace("-", "_")


# ===========================
# FiftyOne Dataset Helpers
# ===========================

def serialize_view(view):
    """Serialize a FiftyOne view"""
    return json.loads(json_util.dumps(view._serialize()))


def get_gt_field(ctx, dataset):
    """Get ground truth field name"""
    if "gt_field" in ctx.params and ctx.params["gt_field"] is not None:
        return ctx.params["gt_field"]
    elif "ground_truth" in dataset.get_field_schema():
        return "ground_truth"
    else:
        return None


def is_subset_view(sample_collection):
    """Check if the sample collection is the entire dataset or a view"""
    return sample_collection.view() != sample_collection._dataset.view()


# ===========================
# FiftyOne <-> W&B Integration
# ===========================

def connect_predictions_to_run(ctx, dataset, predictions_field, project_name, run):
    """Link predictions field to W&B run"""
    # Add run info to predictions field
    field = dataset.get_field(predictions_field)
    field.info = {
        "wandb_project": project_name,
        "wandb_run_name": run.name,
        "wandb_run_id": run.id,
        "wandb_url": run.url,
    }
    field.save()

    # Add tags to W&B run
    try:
        run.tags = list(run.tags) + ["fiftyone", f"predictions:{predictions_field}"]
        run.save()
    except Exception as e:
        print(f"Warning: Could not add tags to W&B run: {e}")

    # Add ground truth field tag
    gt_field = get_gt_field(ctx, dataset)
    if gt_field is not None:
        try:
            run.tags = list(run.tags) + [f"ground_truth:{gt_field}"]
            run.save()
        except Exception as e:
            print(f"Warning: Could not add ground truth tag to W&B run: {e}")


def initialize_fiftyone_run_for_wandb_project(ctx, dataset, project_name):
    """Initialize FiftyOne run for W&B project"""
    config = dataset.init_run()
    wandb_config = get_wandb_config(ctx)
    
    config.method = "wandb_project"
    config.entity = wandb_config["entity"]
    config.project_name = project_name
    config.project_url = get_project_url(ctx, project_name)
    config.runs = []  # List of associated run IDs
    
    dataset.register_run(project_name, config)


def add_fiftyone_run_for_wandb_run(ctx, dataset, project_name, run, **kwargs):
    """Add W&B run to FiftyOne dataset"""
    config = dataset.init_run()
    wandb_config = get_wandb_config(ctx)
    
    config.method = "wandb_run"
    config.run_name = run.name
    config.run_id = run.id
    config.entity = wandb_config["entity"]
    config.project = project_name
    config.state = run.state  # running, finished, crashed, etc.
    config.created_at = run.created_at
    config.url = run.url
    config.config_params = dict(run.config) if run.config else {}
    config.summary_metrics = dict(run.summary) if run.summary else {}
    config.tags = list(run.tags) if run.tags else []
    config.notes = run.notes if hasattr(run, 'notes') else ""
    
    if "predictions_field" in kwargs:
        config.predictions_field = kwargs["predictions_field"]
    if "gt_field" in kwargs:
        config.gt_field = kwargs["gt_field"]
    
    fmt_run_name = format_run_name(run.name)
    dataset.register_run(fmt_run_name, config)
    
    # Add run to project's run list
    try:
        project_run_info = dataset.get_run_info(project_name)
        project_run_info.config.runs.append(run.name)
        dataset.update_run_config(project_name, project_run_info.config)
    except Exception as e:
        print(f"Warning: Could not update project run list: {e}")


def connect_dataset_to_project_if_necessary(ctx, dataset, project_name):
    """Initialize FiftyOne project if it doesn't exist"""
    if project_name not in dataset.list_runs():
        initialize_fiftyone_run_for_wandb_project(ctx, dataset, project_name)


# ===========================
# Dataset Metadata Extraction
# ===========================

def extract_dataset_metadata(dataset, view=None):
    """Extract metadata from FiftyOne dataset/view for WandB logging"""
    import fiftyone as fo
    
    target = view if view is not None else dataset
    
    metadata = {
        # Basic info
        "fiftyone_dataset_name": dataset.name,
        "fiftyone_dataset_size": len(dataset),
        "fiftyone_view_size": len(target),
        "fiftyone_version": fo.__version__,
        "fiftyone_media_type": dataset.media_type,
        
        # Timestamps
        "dataset_created_at": dataset.created_at.isoformat() if dataset.created_at else None,
        "dataset_last_modified_at": dataset.last_modified_at.isoformat() if dataset.last_modified_at else None,
        
        # Custom info
        "dataset_info": dict(dataset.info) if dataset.info else {},
    }
    
    # Add tag-based splits (check all common variations)
    all_tags = target.distinct("tags")
    for split in ["train", "val", "test", "validation", "training", "testing"]:
        if split in all_tags:
            metadata[f"{split}_samples"] = len(target.match_tags(split))
    
    # Add default classes if present
    if dataset.default_classes:
        metadata["default_classes"] = dataset.default_classes
    
    # Add mask targets if present (for segmentation)
    if hasattr(dataset, 'mask_targets') and dataset.mask_targets:
        metadata["mask_targets"] = dict(dataset.mask_targets)
    
    # Special handling for video datasets
    if dataset.media_type == "video":
        frame_schema = dataset.get_frame_field_schema()
        metadata["has_frame_fields"] = bool(frame_schema)
        if frame_schema:
            metadata["frame_fields"] = list(frame_schema.keys())
    
    return metadata


# ===========================
# Query Helpers
# ===========================

def get_candidate_project_names(ctx):
    """Get list of W&B projects stored in FiftyOne"""
    project_names = [
        r
        for r in ctx.dataset.list_runs()
        if ctx.dataset.get_run_info(r).config.method == "wandb_project"
    ]
    return project_names


def get_candidate_run_names(ctx, project_name):
    """Get list of runs for a W&B project"""
    try:
        project_info = ctx.dataset.get_run_info(project_name)
        project_runs = project_info.config.runs
        return project_runs
    except Exception:
        return []

