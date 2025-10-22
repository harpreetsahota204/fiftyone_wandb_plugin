"""Log FiftyOne View to W&B operator - Enhanced with full label support.

This operator logs a FiftyOne view (dataset or filtered subset) to W&B
as a dataset artifact during training, supporting all FiftyOne label types.
"""

import json
import os
import tempfile
from collections import Counter
from datetime import datetime

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import (
    WANDB_AVAILABLE,
    extract_dataset_metadata,
    is_subset_view,
    serialize_view,
)

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================================
# VALIDATION
# ============================================================================

def _validate_inputs(ctx):
    """Validate inputs upfront - fail fast"""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb not installed. Run: pip install wandb")
    
    # Only project is required (run_id is optional, will be auto-generated)
    if not ctx.params.get("project"):
        raise ValueError(f"Missing required parameter: project")
    
    for secret in ["FIFTYONE_WANDB_API_KEY", "FIFTYONE_WANDB_ENTITY"]:
        if not ctx.secret(secret):
            raise ValueError(f"{secret} not set")


# ============================================================================
# LABEL FORMATTING - Supports ALL FiftyOne label types
# https://docs.voxel51.com/api/fiftyone.core.labels.html
# ============================================================================

def _format_label(label):
    """
    Format any FiftyOne label type for WandB Table display.
    
    Supports all label types from:
    https://docs.voxel51.com/api/fiftyone.core.labels.html
    """
    
    # Classification
    if isinstance(label, fol.Classification):
        conf = f" ({label.confidence:.2f})" if label.confidence else ""
        return f"{label.label}{conf}"
    
    # Detections
    if isinstance(label, fol.Detections):
        if not label.detections:
            return "0 detections"
        
        # Count each label type
        label_counts = Counter(d.label for d in label.detections if d.label)
        
        # Format as "person(3), car(2), dog(1)"
        counts_str = ", ".join(f"{label}({count})" for label, count in label_counts.most_common())
        return f"{len(label.detections)} detections: {counts_str}"
    
    # Detection (single)
    if isinstance(label, fol.Detection):
        conf = f" ({label.confidence:.2f})" if label.confidence else ""
        return f"{label.label}{conf}"
    
    # Polyline
    if isinstance(label, fol.Polyline):
        label_str = label.label or "polyline"
        points_count = len(label.points[0]) if label.points else 0
        return f"{label_str} ({points_count} points)"
    
    # Polylines
    if isinstance(label, fol.Polylines):
        if not label.polylines:
            return "0 polylines"
        
        # Count each polyline type
        label_counts = Counter(p.label for p in label.polylines if p.label)
        
        # Format as "road(2), sidewalk(1)"
        counts_str = ", ".join(f"{label}({count})" for label, count in label_counts.most_common())
        return f"{len(label.polylines)} polylines: {counts_str}"
    
    # Keypoint
    if isinstance(label, fol.Keypoint):
        return label.label or "keypoint"
    
    # Keypoints
    if isinstance(label, fol.Keypoints):
        if not label.keypoints:
            return "0 keypoints"
        
        # Count each keypoint type
        label_counts = Counter(k.label for k in label.keypoints if k.label)
        
        # Format as "nose(1), left_eye(1), right_eye(1)"
        counts_str = ", ".join(f"{label}({count})" for label, count in label_counts.most_common())
        return f"{len(label.keypoints)} keypoints: {counts_str}"
    
    # Segmentation
    if isinstance(label, fol.Segmentation):
        return "segmentation mask"
    
    # Heatmap
    if isinstance(label, fol.Heatmap):
        range_str = f" [{label.range[0]:.2f}, {label.range[1]:.2f}]" if label.range else ""
        return f"heatmap{range_str}"
    
    # TemporalDetection
    if isinstance(label, fol.TemporalDetection):
        conf = f" ({label.confidence:.2f})" if label.confidence else ""
        return f"{label.label}{conf} @ {label.support}"
    
    # TemporalDetections
    if isinstance(label, fol.TemporalDetections):
        if not label.detections:
            return "0 temporal detections"
        
        # Count each temporal detection type
        label_counts = Counter(d.label for d in label.detections if d.label)
        
        # Format as "action(2), speaking(1)"
        counts_str = ", ".join(f"{label}({count})" for label, count in label_counts.most_common())
        return f"{len(label.detections)} temporal detections: {counts_str}"
    
    # GeoLocation
    if isinstance(label, fol.GeoLocation):
        if label.point:
            return f"point {label.point}"
        elif label.line:
            return f"line ({len(label.line)} points)"
        elif label.polygon:
            return f"polygon ({len(label.polygon[0])} points)"
        return "geolocation"
    
    # GeoLocations
    if isinstance(label, fol.GeoLocations):
        total = len(label.points or []) + len(label.lines or []) + len(label.polygons or [])
        return f"{total} geolocations"
    
    # Regression
    if isinstance(label, fol.Regression):
        conf = f" ({label.confidence:.2f})" if label.confidence else ""
        return f"{label.value}{conf}"
    
    # Fallback for unknown types
    if hasattr(label, 'label'):
        return str(label.label)
    
    return str(type(label).__name__)


def _get_label_fields(view):
    """Get all label field names from view schema"""
    label_fields = []
    schema = view.get_field_schema()
    
    for field_name, field in schema.items():
        # Check if it's a label field by looking at the document type
        if hasattr(field, 'document_type'):
            doc_type = field.document_type
            # Check if it inherits from Label
            if issubclass(doc_type, fol.Label):
                label_fields.append(field_name)
    
    return label_fields


# ============================================================================
# DATA COLLECTION
# ============================================================================

def _add_labels_table(artifact, view):
    """Add labels as WandB Table with support for all label types"""
    
    # Build table columns
    columns = ["sample_id", "filepath", "tags"]
    label_fields = _get_label_fields(view)
    
    # Add column for each label field
    for field in label_fields:
        columns.append(f"label_{field}")
    
    # Add image column for visualization
    columns.append("image")
    
    # Collect rows
    rows = []
    for sample in view.iter_samples(progress=True):
        row = [
            sample.id,
            sample.filepath,
            ",".join(sample.tags or []),
        ]
        
        # Add each label field
        for field in label_fields:
            label = sample[field] if field in sample else None
            row.append(_format_label(label) if label else None)
        
        # Add image
        try:
            row.append(wandb.Image(sample.filepath))
        except Exception:
            row.append(None)
        
        rows.append(row)
    
    # Create and add table
    table = wandb.Table(columns=columns, data=rows)
    artifact.add(table, "samples")


def _add_sample_references(artifact, view):
    """Add lightweight sample references (no labels)"""
    
    sample_refs = []
    for sample in view.iter_samples(progress=True):
        ref = {
            "id": sample.id,
            "filepath": sample.filepath,
            "tags": sample.tags or [],
        }
        if sample.metadata:
            ref["metadata"] = dict(sample.metadata)
        sample_refs.append(ref)
    
    # Use context manager for automatic cleanup
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_refs, f, indent=2)
        temp_path = f.name
    
    try:
        artifact.add_file(temp_path, name="sample_references.json")
    finally:
        os.unlink(temp_path)


def _add_embeddings(artifact, view, embedding_field):
    """Add embeddings to artifact - simple numpy file"""
    import numpy as np
    
    # Collect embeddings
    sample_ids = []
    embeddings = []
    
    for sample in view.iter_samples(progress=True):
        emb = sample[embedding_field] if embedding_field in sample else None
        if emb is not None:
            sample_ids.append(sample.id)
            embeddings.append(emb)
    
    if not embeddings:
        return  # No embeddings found, skip
    
    # Save as compressed numpy
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez_compressed(
            f,
            sample_ids=sample_ids,
            embeddings=np.array(embeddings)
        )
        temp_path = f.name
    
    try:
        artifact.add_file(temp_path, name=f"embeddings/{embedding_field}.npz")
    finally:
        os.unlink(temp_path)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def _log_fiftyone_view_to_wandb(ctx):
    """Log FiftyOne view to WandB with comprehensive label support"""
    
    # 1. Validate
    _validate_inputs(ctx)
    
    # 2. Prepare
    view = ctx.target_view()  # Use target_view to support selected samples
    dataset = ctx.dataset
    project_name = ctx.params.get("project")
    run_id = ctx.params.get("run_id")
    
    # Auto-generate run_id if not provided (for UI usage)
    if not run_id:
        import uuid
        run_id = f"fiftyone_{uuid.uuid4().hex[:8]}"
    
    # Get and sanitize artifact name
    artifact_name = ctx.params.get("artifact_name")
    if artifact_name:
        # Sanitize: lowercase, replace spaces and invalid chars with hyphens
        import re
        artifact_name = artifact_name.lower()
        artifact_name = re.sub(r'[^a-z0-9\-_.]', '-', artifact_name)
        artifact_name = re.sub(r'-+', '-', artifact_name)  # Remove multiple hyphens
        artifact_name = artifact_name.strip('-')  # Remove leading/trailing hyphens
    else:
        artifact_name = f"dataset_view_{run_id[:8]}"
    
    # 3. Create artifact
    metadata = extract_dataset_metadata(dataset, view)
    if is_subset_view(view):
        metadata["view_serialization"] = serialize_view(view)
        metadata["is_subset_view"] = True
    else:
        metadata["is_subset_view"] = False
    
    # Add sample IDs for easy view recreation
    sample_ids = view.values("id")
    metadata["sample_ids"] = sample_ids
    metadata["num_samples"] = len(sample_ids)
    
    artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        description=f"Training view for run {run_id}",
        metadata=metadata
    )
    
    # 4. Add data (labels or lightweight references)
    if ctx.params.get("include_labels", False):
        _add_labels_table(artifact, view)
    else:
        _add_sample_references(artifact, view)
    
    # Optional: add embeddings
    if ctx.params.get("include_embeddings", False):
        embedding_field = ctx.params.get("embedding_field")
        if embedding_field:
            _add_embeddings(artifact, view, embedding_field)
    
    # 5. Upload to WandB
    entity = ctx.secret("FIFTYONE_WANDB_ENTITY")
    
    # Use resume="allow" to handle both existing and new runs
    # If run exists, it resumes; if not, it creates new
    with wandb.init(project=project_name, id=run_id, resume="allow", entity=entity) as run:
        run.log_artifact(artifact)
        run.config.update({
            "fiftyone_view_artifact": f"{artifact_name}:latest",
            "fiftyone_dataset_name": dataset.name,
            "fiftyone_view_size": len(view),
            "fiftyone_is_subset": is_subset_view(view),
        })
        wandb_url = run.url
    
    # 6. Register in FiftyOne
    run_config = dataset.init_run()
    run_config.method = "wandb_training"
    run_config.wandb_run_id = run_id
    run_config.wandb_project = project_name
    run_config.dataset_artifact = f"{artifact_name}:latest"
    run_config.samples_used = len(view)
    if is_subset_view(view):
        run_config.view_serialization = serialize_view(view)
    
    # Run keys must be valid Python variable names (no hyphens, no special chars)
    run_key = f"dataset_view_{run_id}".replace("-", "_").replace(".", "_")
    dataset.register_run(run_key, run_config)
    
    return {
        "success": True,
        "artifact_name": artifact_name,
        "samples_logged": len(view),
        "wandb_url": wandb_url,
    }


# ============================================================================
# OPERATOR
# ============================================================================

class LogFiftyOneViewToWandB(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="log_fiftyone_view_to_wandb",
            label="W&B: Save View as Artifact",
            description="Save FiftyOne view to WandB as dataset artifact",
            dynamic=True,
            icon="/assets/wandb.svg",
            execution_options=foo.ExecutionOptions(
                allow_immediate=True,
                allow_delegation=True,
            ),
        )
    
    def __call__(
        self,
        sample_collection,
        project,
        run_id,
        artifact_name=None,
        include_labels=False,
    ):
        """Programmatic interface for logging FiftyOne views to WandB"""
        dataset = sample_collection._dataset
        view = sample_collection.view()
        
        class MockContext:
            def __init__(self, view, dataset, params):
                self.view = view
                self.dataset = dataset
                self.params = params
            
            def secret(self, name):
                return os.environ.get(name)
        
        ctx = MockContext(view, dataset, {
            "project": project,
            "run_id": run_id,
            "artifact_name": artifact_name,
            "include_labels": include_labels,
        })
        
        return _log_fiftyone_view_to_wandb(ctx)
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Add view target selector (Dataset, Current view, or Selected samples)
        inputs.view_target(ctx)
        
        # Warning about full dataset
        target = ctx.target_view()
        if not is_subset_view(target):
            inputs.view("warning", types.Warning(
                label="Logging full dataset",
                description=f"Consider creating a filtered view. Current: {len(ctx.dataset)} samples"
            ))
        
        # Basic inputs
        inputs.str("project", label="W&B Project", required=True, 
                   default=ctx.secret("FIFTYONE_WANDB_PROJECT"))
        inputs.str("run_id", label="W&B Run ID (optional)", required=False,
                   description="From wandb.run.id in your training script. Auto-generated if not provided. Use lowercase, numbers, hyphens, underscores only. No spaces or special chars.")
        inputs.str("artifact_name", label="Artifact Name (optional)", required=False,
                   description="Use lowercase, numbers, hyphens, underscores only. No spaces or special chars.")
        
        # Label logging option
        inputs.bool("include_labels", label="Include Labels", default=False,
                    description="Log labels to WandB Table (slower but more useful)")
        
        # Show label fields that will be included
        if ctx.params.get("include_labels"):
            label_fields = _get_label_fields(ctx.target_view())
            if label_fields:
                inputs.view("label_info", types.Notice(
                    label=f"Will log {len(label_fields)} label fields",
                    description=f"Fields: {', '.join(label_fields)}"
                ))
        
        # Embeddings option (only if labels are included)
        if ctx.params.get("include_labels"):
            emb_fields = [f for f in ctx.dataset.get_field_schema().keys() 
                          if "embedding" in f.lower() or "vector" in f.lower()]
            
            if emb_fields:
                inputs.bool("include_embeddings", label="Include Embeddings", default=False)
                
                if ctx.params.get("include_embeddings"):
                    emb_choices = [types.Choice(label=f, value=f) for f in emb_fields]
                    inputs.enum("embedding_field", [c.value for c in emb_choices],
                               label="Embedding Field",
                               view=types.AutocompleteView(choices=emb_choices))
        
        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        """Delegate for large datasets with labels"""
        include_labels = ctx.params.get("include_labels", False)
        return include_labels and len(ctx.target_view()) > 1000
    
    def execute(self, ctx):
        return _log_fiftyone_view_to_wandb(ctx)
    
    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("artifact_name", label="Artifact Name")
        outputs.int("samples_logged", label="Samples Logged")
        outputs.str("wandb_url", label="WandB Run URL")
        return types.Property(outputs)

