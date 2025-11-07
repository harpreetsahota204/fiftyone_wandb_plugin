"""Log Model Predictions to W&B operator.

This operator logs model predictions from FiftyOne to WandB as a predictions artifact,
enabling model output tracking, analysis, and comparison across experiments.
"""

import json
import os
import uuid
from collections import Counter
from datetime import datetime

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import WANDB_AVAILABLE

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
    
    if not ctx.params.get("project"):
        raise ValueError("Missing required parameter: project")
    
    if not ctx.params.get("model_name"):
        raise ValueError("Missing required parameter: model_name")
    
    if not ctx.params.get("predictions_field"):
        raise ValueError("Missing required parameter: predictions_field")
    
    # Check secrets with fallback to environment variables
    for secret in ["FIFTYONE_WANDB_API_KEY", "FIFTYONE_WANDB_ENTITY"]:
        value = ctx.secrets.get(secret) or os.getenv(secret)
        if not value:
            raise ValueError(
                f"{secret} not set. Set as environment variable: export {secret}='value'"
            )


# ============================================================================
# LABEL ID COLLECTION
# ============================================================================

def _collect_label_ids(label):
    """Extract all label IDs from a label object"""
    if isinstance(label, fol.Detections):
        return [d.id for d in label.detections]
    elif isinstance(label, fol.Polylines):
        return [p.id for p in label.polylines]
    elif isinstance(label, fol.Keypoints):
        return [k.id for k in label.keypoints]
    elif isinstance(label, fol.TemporalDetections):
        return [d.id for d in label.detections]
    elif isinstance(label, fol.GeoLocations):
        ids = []
        if label.points:
            ids.extend([p.id for p in label.points if hasattr(p, 'id')])
        if label.lines:
            ids.extend([l.id for l in label.lines if hasattr(l, 'id')])
        if label.polygons:
            ids.extend([p.id for p in label.polygons if hasattr(p, 'id')])
        return ids
    elif hasattr(label, 'id'):
        return [label.id]
    return []


# ============================================================================
# STATISTICS CALCULATION
# ============================================================================

def _get_avg_confidence(label):
    """Get average confidence from predictions"""
    if isinstance(label, fol.Detections):
        confs = [d.confidence for d in label.detections if d.confidence]
        return round(sum(confs) / len(confs), 3) if confs else None
    elif isinstance(label, fol.Polylines):
        confs = [p.confidence for p in label.polylines if hasattr(p, 'confidence') and p.confidence]
        return round(sum(confs) / len(confs), 3) if confs else None
    elif isinstance(label, fol.Keypoints):
        confs = [k.confidence for k in label.keypoints if hasattr(k, 'confidence') and k.confidence]
        return round(sum(confs) / len(confs), 3) if confs else None
    elif hasattr(label, 'confidence'):
        return round(label.confidence, 3) if label.confidence else None
    return None


def _get_num_predictions(label):
    """Count predictions in label"""
    if isinstance(label, fol.Detections):
        return len(label.detections)
    elif isinstance(label, fol.Polylines):
        return len(label.polylines)
    elif isinstance(label, fol.Keypoints):
        return len(label.keypoints)
    elif isinstance(label, fol.TemporalDetections):
        return len(label.detections)
    elif isinstance(label, fol.GeoLocations):
        total = 0
        if label.points:
            total += len(label.points)
        if label.lines:
            total += len(label.lines)
        if label.polygons:
            total += len(label.polygons)
        return total
    return 1 if label else 0


def _format_label(label):
    """
    Format any FiftyOne label type for WandB Table display.
    Reuses the same logic from log_fiftyone_view_to_wandb
    """
    # Classification
    if isinstance(label, fol.Classification):
        conf = f" ({label.confidence:.2f})" if label.confidence else ""
        return f"{label.label}{conf}"
    
    # Detections
    if isinstance(label, fol.Detections):
        if not label.detections:
            return "0 detections"
        
        label_counts = Counter(d.label for d in label.detections if d.label)
        counts_str = ", ".join(f"{lbl}({count})" for lbl, count in label_counts.most_common())
        return f"{len(label.detections)} detections: {counts_str}"
    
    # Detection (single)
    if isinstance(label, fol.Detection):
        conf = f" ({label.confidence:.2f})" if label.confidence else ""
        return f"{label.label}{conf}"
    
    # Polylines
    if isinstance(label, fol.Polylines):
        if not label.polylines:
            return "0 polylines"
        label_counts = Counter(p.label for p in label.polylines if p.label)
        counts_str = ", ".join(f"{lbl}({count})" for lbl, count in label_counts.most_common())
        return f"{len(label.polylines)} polylines: {counts_str}"
    
    # Keypoints
    if isinstance(label, fol.Keypoints):
        if not label.keypoints:
            return "0 keypoints"
        label_counts = Counter(k.label for k in label.keypoints if k.label)
        counts_str = ", ".join(f"{lbl}({count})" for lbl, count in label_counts.most_common())
        return f"{len(label.keypoints)} keypoints: {counts_str}"
    
    # TemporalDetections
    if isinstance(label, fol.TemporalDetections):
        if not label.detections:
            return "0 temporal detections"
        label_counts = Counter(d.label for d in label.detections if d.label)
        counts_str = ", ".join(f"{lbl}({count})" for lbl, count in label_counts.most_common())
        return f"{len(label.detections)} temporal detections: {counts_str}"
    
    # Segmentation
    if isinstance(label, fol.Segmentation):
        return "segmentation mask"
    
    # Heatmap
    if isinstance(label, fol.Heatmap):
        range_str = f" [{label.range[0]:.2f}, {label.range[1]:.2f}]" if label.range else ""
        return f"heatmap{range_str}"
    
    # Fallback
    if hasattr(label, 'label'):
        return str(label.label)
    
    return str(type(label).__name__)


def _calculate_class_distribution(view, pred_field):
    """Calculate overall class distribution from predictions"""
    all_classes = []
    
    for sample in view.iter_samples():
        pred = sample[pred_field] if pred_field in sample else None
        
        if isinstance(pred, fol.Detections):
            all_classes.extend(d.label for d in pred.detections if d.label)
        elif isinstance(pred, fol.Polylines):
            all_classes.extend(p.label for p in pred.polylines if p.label)
        elif isinstance(pred, fol.Keypoints):
            all_classes.extend(k.label for k in pred.keypoints if k.label)
        elif isinstance(pred, fol.TemporalDetections):
            all_classes.extend(d.label for d in pred.detections if d.label)
        elif hasattr(pred, 'label') and pred.label:
            all_classes.append(pred.label)
    
    return dict(Counter(all_classes))


# ============================================================================
# TABLE CREATION
# ============================================================================

def _create_predictions_table(view, pred_field, prompt_field=None, include_images=False):
    """Create predictions table with optional per-sample prompts"""
    
    columns = ["sample_id", "label_ids"]
    
    # Add prompt column if field specified
    if prompt_field:
        columns.append("prompt")
    
    columns.extend(["predictions", "avg_confidence", "num_predictions"])
    
    if include_images:
        columns.append("image")
    
    all_label_ids = []
    low_confidence_label_ids = []
    rows = []
    
    for sample in view.iter_samples(progress=True):
        pred_label = sample[pred_field] if pred_field in sample else None
        
        # Collect label IDs
        sample_label_ids = _collect_label_ids(pred_label)
        all_label_ids.extend(sample_label_ids)
        
        # Track low confidence labels (for active learning) - works for ALL label types
        if isinstance(pred_label, fol.Detections):
            for det in pred_label.detections:
                if det.confidence and det.confidence < 0.5:
                    low_confidence_label_ids.append(det.id)
        elif isinstance(pred_label, fol.Polylines):
            for poly in pred_label.polylines:
                if hasattr(poly, 'confidence') and poly.confidence and poly.confidence < 0.5:
                    low_confidence_label_ids.append(poly.id)
        elif isinstance(pred_label, fol.Keypoints):
            for kp in pred_label.keypoints:
                if hasattr(kp, 'confidence') and kp.confidence and kp.confidence < 0.5:
                    low_confidence_label_ids.append(kp.id)
        elif isinstance(pred_label, fol.TemporalDetections):
            for det in pred_label.detections:
                if det.confidence and det.confidence < 0.5:
                    low_confidence_label_ids.append(det.id)
        elif hasattr(pred_label, 'confidence') and hasattr(pred_label, 'id'):
            # Single label types (Classification, Detection, Regression, etc.)
            if pred_label.confidence and pred_label.confidence < 0.5:
                low_confidence_label_ids.append(pred_label.id)
        
        row = [
            sample.id,
            sample_label_ids,
        ]
        
        # Add per-sample prompt if specified
        if prompt_field:
            prompt = sample[prompt_field] if prompt_field in sample else None
            row.append(prompt)
        
        row.extend([
            _format_label(pred_label) if pred_label else None,
            _get_avg_confidence(pred_label),
            _get_num_predictions(pred_label),
        ])
        
        if include_images:
            try:
                row.append(wandb.Image(sample.filepath))
            except Exception:
                row.append(None)
        
        rows.append(row)
    
    table = wandb.Table(columns=columns, data=rows)
    return table, all_label_ids, low_confidence_label_ids


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def _log_model_predictions(ctx):
    """Log model predictions to WandB"""
    
    # 1. Validate
    _validate_inputs(ctx)
    
    # 2. Prepare
    view = ctx.target_view()
    dataset = ctx.dataset
    project_name = ctx.params.get("project")
    model_name = ctx.params.get("model_name")
    model_version = ctx.params.get("model_version", "")
    pred_field = ctx.params.get("predictions_field")
    prompt_field = ctx.params.get("prompt_field")
    include_images = ctx.params.get("include_images", False)
    
    # Parse model_config (could be dict from __call__ or JSON string from UI)
    model_config = ctx.params.get("model_config", {})
    if isinstance(model_config, str):
        # Parse JSON string from UI
        try:
            model_config = json.loads(model_config) if model_config.strip() else {}
        except json.JSONDecodeError:
            model_config = {"raw_config": model_config}  # Fallback
    elif model_config is None:
        model_config = {}
    
    # Also check for model_config_json from UI
    model_config_json = ctx.params.get("model_config_json")
    if model_config_json and not model_config:
        try:
            model_config = json.loads(model_config_json) if model_config_json.strip() else {}
        except json.JSONDecodeError:
            model_config = {"raw_config": model_config_json}
    
    # Auto-generate artifact name
    artifact_name = ctx.params.get("artifact_name")
    if artifact_name:
        # Sanitize
        import re
        artifact_name = artifact_name.lower()
        artifact_name = re.sub(r'[^a-z0-9\-_.]', '-', artifact_name)
        artifact_name = re.sub(r'-+', '-', artifact_name)
        artifact_name = artifact_name.strip('-')
    else:
        # Auto-generate: model_name_predictions_timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_name = f"{model_name.lower()}_predictions_{timestamp}"
        artifact_name = artifact_name.replace(" ", "_")
    
    # 3. Create predictions table
    table, all_label_ids, low_confidence_ids = _create_predictions_table(
        view, pred_field, prompt_field, include_images
    )
    
    # 4. Calculate statistics
    class_dist = _calculate_class_distribution(view, pred_field)
    sample_ids = view.values("id")
    
    # 5. Create artifact metadata (model info goes HERE, not in table!)
    metadata = {
        # Model information
        "model_name": model_name,
        "model_version": model_version,
        "model_config": model_config,  # Global config (system prompt, temp, etc.)
        "predictions_field": pred_field,
        "prompt_field": prompt_field,  # Field containing per-sample prompts (if any)
        
        # Dataset information
        "fiftyone_dataset_name": dataset.name,
        "fiftyone_dataset_size": len(dataset),
        "fiftyone_view_size": len(view),
        
        # Sample and label tracking
        "sample_ids": sample_ids,
        "label_ids": all_label_ids,
        "num_samples": len(sample_ids),
        "total_predictions": len(all_label_ids),
        
        # Statistics
        "avg_predictions_per_sample": round(len(all_label_ids) / len(sample_ids), 2) if sample_ids else 0,
        "class_distribution": class_dist,
        "num_classes": len(class_dist),
        
        # Metadata for workflows
        "inference_timestamp": datetime.now().isoformat(),
        "low_confidence_label_ids": low_confidence_ids,
        "num_low_confidence": len(low_confidence_ids),
    }
    
    # 6. Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="predictions",
        description=f"{model_name} predictions on {dataset.name}",
        metadata=metadata
    )
    
    # Add table
    artifact.add(table, "predictions")
    
    # 7. Upload to WandB
    entity = ctx.secrets.get("FIFTYONE_WANDB_ENTITY") or os.getenv("FIFTYONE_WANDB_ENTITY")
    api_key = ctx.secrets.get("FIFTYONE_WANDB_API_KEY") or os.getenv("FIFTYONE_WANDB_API_KEY")
    
    # Login to WandB first
    if api_key:
        wandb.login(key=api_key)
    
    run_id = f"predictions_{uuid.uuid4().hex[:8]}"
    
    with wandb.init(project=project_name, id=run_id, resume="allow", entity=entity) as run:
        run.log_artifact(artifact)
        run.config.update({
            "model_name": model_name,
            "model_version": model_version,
            "predictions_field": pred_field,
            "samples_processed": len(view),
        })
        wandb_url = run.url
    
    # 8. Register in FiftyOne
    run_config = dataset.init_run()
    run_config.method = "model_inference"
    run_config.model_name = model_name
    run_config.model_version = model_version
    run_config.predictions_field = pred_field
    run_config.wandb_project = project_name
    run_config.wandb_artifact = f"{artifact_name}:latest"
    run_config.samples_processed = len(view)
    run_config.total_predictions = len(all_label_ids)
    run_config.inference_timestamp = datetime.now().isoformat()
    
    # Sanitize run key
    run_key = f"inference_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_key = run_key.replace("-", "_").replace(".", "_").replace(" ", "_")
    dataset.register_run(run_key, run_config)
    
    return {
        "success": True,
        "artifact_name": artifact_name,
        "samples_processed": len(view),
        "total_predictions": len(all_label_ids),
        "num_classes": len(class_dist),
        "wandb_url": wandb_url,
    }


# ============================================================================
# OPERATOR
# ============================================================================

class LogModelPredictions(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="log_model_predictions",
            label="W&B: Log Model Predictions",
            description="Log model predictions to WandB for tracking and analysis",
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
        model_name,
        predictions_field,
        project,
        model_version=None,
        model_config=None,
        prompt_field=None,
        artifact_name=None,
        include_images=False,
    ):
        """
        Programmatic interface for logging model predictions to WandB
        
        Args:
            sample_collection: FiftyOne dataset or view
            model_name: Name of the model (required)
            predictions_field: Field containing predictions (required)
            project: WandB project name (required)
            model_version: Model version string (optional)
            model_config: Global model configuration dict (optional)
                Example: {"system_prompt": "...", "temperature": 0.7}
            prompt_field: Field name containing per-sample prompts (optional)
            artifact_name: Custom artifact name (optional)
            include_images: Include thumbnail images (default: False)
        """
        dataset = sample_collection._dataset
        view = sample_collection.view()
        
        class MockContext:
            def __init__(self, view, dataset, params):
                self.view = view
                self.dataset = dataset
                self.params = params
                self._target_view = view
            
            @property
            def secrets(self):
                # Return a dict-like object that falls back to os.environ
                class SecretsDict(dict):
                    def get(self, key, default=None):
                        return os.getenv(key, default)
                return SecretsDict()
            
            def target_view(self):
                return self._target_view
        
        ctx = MockContext(view, dataset, {
            "project": project,
            "model_name": model_name,
            "model_version": model_version,
            "model_config": model_config,
            "predictions_field": predictions_field,
            "prompt_field": prompt_field,
            "artifact_name": artifact_name,
            "include_images": include_images,
        })
        
        return _log_model_predictions(ctx)
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Target view selector (Dataset, Current view, or Selected samples)
        inputs.view_target(ctx)
        
        # Model information
        inputs.str(
            "model_name",
            label="Model Name",
            description="Name of the model (e.g., 'yolov8n', 'resnet50')",
            required=True
        )
        
        inputs.str(
            "model_version",
            label="Model Version (optional)",
            description="Version or variant (e.g., 'v1.0', 'large')",
            required=False
        )
        
        # Predictions field (auto-detect from schema)
        pred_fields = [
            f for f in ctx.dataset.get_field_schema().keys() 
            if any(keyword in f.lower() for keyword in ["pred", "prediction", "inference", "output"])
        ]
        
        if pred_fields:
            pred_choices = [types.Choice(label=f, value=f) for f in pred_fields]
            inputs.enum(
                "predictions_field",
                [c.value for c in pred_choices],
                label="Predictions Field",
                description="Field containing model predictions",
                required=True,
                view=types.AutocompleteView(choices=pred_choices)
            )
        else:
            inputs.str(
                "predictions_field",
                label="Predictions Field",
                description="Field containing model predictions",
                required=True
            )
        
        # WandB project
        inputs.str(
            "project",
            label="W&B Project",
            description="WandB project for logging predictions",
            required=True,
            default=ctx.secrets.get("FIFTYONE_WANDB_PROJECT")
        )
        
        # Artifact name (optional)
        inputs.str(
            "artifact_name",
            label="Artifact Name (optional)",
            description="Auto-generated if not provided. Use lowercase, numbers, hyphens, underscores only.",
            required=False
        )
        
        # Advanced options
        inputs.view(
            "advanced_header",
            types.Header(
                label="Advanced Options",
                divider=True
            )
        )
        
        # Prompt field (for per-sample prompts)
        prompt_fields = [
            f for f in ctx.dataset.get_field_schema().keys()
            if any(keyword in f.lower() for keyword in ["prompt", "instruction", "query", "text"])
        ]
        
        if prompt_fields:
            prompt_choices = [types.Choice(label=f, value=f) for f in prompt_fields]
            inputs.enum(
                "prompt_field",
                [c.value for c in prompt_choices],
                label="Prompt Field (optional)",
                description="Field containing sample-specific prompts (e.g., for VLMs)",
                view=types.AutocompleteView(choices=prompt_choices)
            )
        
        # Model config JSON (for global config like system prompt, temperature)
        inputs.str(
            "model_config_json",
            label="Model Config JSON (optional)",
            description="Global model configuration: system prompt, temperature, etc.",
            view=types.CodeView(language="json")
        )
        
        # Include images option (default: NO - keep it lightweight!)
        inputs.bool(
            "include_images",
            label="Include Thumbnail Images",
            description="⚠️ Warning: Significantly increases artifact size. Leave OFF for large datasets.",
            default=False
        )
        
        # Show stats about what will be logged
        target = ctx.target_view()
        info_desc = f"Predictions from field: '{ctx.params.get('predictions_field', 'N/A')}'"
        if ctx.params.get("prompt_field"):
            info_desc += f"\nSample prompts from: '{ctx.params.get('prompt_field')}'"
        
        inputs.view(
            "info",
            types.Notice(
                label=f"Will process {len(target)} samples",
                description=info_desc
            )
        )
        
        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        """Delegate for large datasets with images"""
        include_images = ctx.params.get("include_images", False)
        # Only delegate if including images and dataset is large
        return include_images and len(ctx.target_view()) > 500
    
    def execute(self, ctx):
        return _log_model_predictions(ctx)
    
    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("artifact_name", label="Artifact Name")
        outputs.int("samples_processed", label="Samples Processed")
        outputs.int("total_predictions", label="Total Predictions")
        outputs.int("num_classes", label="Number of Classes")
        outputs.str("wandb_url", label="WandB URL", view=types.LinkView())
        return types.Property(outputs)

