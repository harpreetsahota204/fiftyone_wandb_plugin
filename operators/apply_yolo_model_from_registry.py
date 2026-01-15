"""Apply YOLO Model from W&B Registry operator.

This operator allows users to select a YOLO model from the Weights & Biases
Model Registry and apply it to their FiftyOne dataset for inference.
"""

import os
import uuid
from collections import Counter
from datetime import datetime

import fiftyone.core.labels as fol
import fiftyone.operators as foo
import fiftyone.operators.types as types
import wandb

from ..wandb_helpers import (
    create_mock_context,
    ensure_wandb_login,
    get_credentials,
    get_run_url,
    get_wandb_api,
    prompt_for_missing_credentials,
    sanitize_for_artifact,
    sanitize_for_run_key,
)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

# Local cache directory for downloaded models
MODEL_CACHE_DIR = "/tmp/wandb_model_cache"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _ensure_cache_dir():
    """Create model cache directory if it doesn't exist."""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return MODEL_CACHE_DIR


def _collect_label_ids(label):
    """Extract all label IDs from a label object."""
    if isinstance(label, fol.Detections):
        return [d.id for d in label.detections]
    elif isinstance(label, fol.Classification):
        return [label.id] if hasattr(label, 'id') else []
    elif isinstance(label, fol.Polylines):
        return [p.id for p in label.polylines]
    elif isinstance(label, fol.Keypoints):
        return [k.id for k in label.keypoints]
    elif hasattr(label, 'id'):
        return [label.id]
    return []


def _calculate_class_distribution(view, pred_field):
    """Calculate class distribution from predictions."""
    all_classes = []
    
    for sample in view.iter_samples():
        pred = sample[pred_field] if pred_field in sample else None
        if isinstance(pred, fol.Detections):
            all_classes.extend(d.label for d in pred.detections if d.label)
        elif isinstance(pred, fol.Classification):
            if pred.label:
                all_classes.append(pred.label)
        elif hasattr(pred, 'label') and pred.label:
            all_classes.append(pred.label)
    
    return dict(Counter(all_classes))


def _get_model_artifacts(api, entity, project_name):
    """Get all model artifacts from a W&B project.
    
    Args:
        api: W&B API client
        entity: W&B entity (username or team)
        project_name: W&B project name
        
    Returns:
        list: List of dicts with artifact info (name, full_name, metadata)
    """
    model_artifacts = []
    collections = api.artifact_collections(
        project_name=f"{entity}/{project_name}",
        type_name="model"
    )
    for collection in collections:
        # Get all versions of this model
        for artifact in collection.artifacts():
            model_artifacts.append({
                "name": artifact.name,  # e.g., "my_model:v0"
                "full_name": f"{entity}/{project_name}/{artifact.name}",
                "metadata": artifact.metadata or {},
                "description": artifact.description or "",
                "created_at": artifact.created_at,
            })
    
    return model_artifacts


def _download_model_from_artifact(api, artifact_path, cache_dir=None):
    """Download model weights from a W&B artifact.
    
    Args:
        api: W&B API client
        artifact_path: Full artifact path (entity/project/name:version)
        cache_dir: Directory to download to (default: MODEL_CACHE_DIR)
        
    Returns:
        tuple: (local_weights_path, artifact_metadata)
    """
    if cache_dir is None:
        cache_dir = _ensure_cache_dir()
    
    # Fetch the artifact
    artifact = api.artifact(artifact_path)
    
    # Create a unique subdirectory for this artifact version
    artifact_dir = os.path.join(cache_dir, artifact.name.replace(":", "_"))
    
    # Download artifact files
    artifact.download(root=artifact_dir)
    
    # Find the weights file (typically best.pt for YOLO models)
    weights_candidates = ["best.pt", "last.pt", "model.pt", "weights.pt"]
    weights_path = None
    
    for candidate in weights_candidates:
        candidate_path = os.path.join(artifact_dir, candidate)
        if os.path.exists(candidate_path):
            weights_path = candidate_path
            break
    
    # If no standard name found, look for any .pt file
    if weights_path is None:
        for file in os.listdir(artifact_dir):
            if file.endswith(".pt"):
                weights_path = os.path.join(artifact_dir, file)
                break
    
    if weights_path is None:
        raise FileNotFoundError(
            f"No .pt weights file found in artifact {artifact_path}. "
            f"Files in artifact: {os.listdir(artifact_dir)}"
        )
    
    return weights_path, artifact.metadata or {}


def _apply_yolo_model(ctx):
    """Apply YOLO model from W&B registry to dataset."""
    # Ensure W&B is logged in
    ensure_wandb_login(ctx)
    
    # Get parameters
    entity, _, _ = get_credentials(ctx)
    
    project_name = ctx.params["project"]
    model_artifact = ctx.params["model_artifact"]
    predictions_field = ctx.params["predictions_field"]
    confidence_threshold = ctx.params.get("confidence_threshold", 0.25)
    log_to_wandb = ctx.params.get("log_to_wandb", False)
    run_name = ctx.params.get("run_name")
    predictions_artifact_name = ctx.params.get("predictions_artifact_name")
    
    # Get target view
    target = ctx.target_view()
    dataset = ctx.dataset
    
    # Get W&B API
    api = get_wandb_api(ctx)
    
    # Download model weights from artifact
    weights_path, artifact_metadata = _download_model_from_artifact(
        api, model_artifact
    )
    
    # Extract model info from metadata
    task_type = artifact_metadata.get("task_type", "detection")
    model_name = artifact_metadata.get("model_name", "unknown")
    
    # Load YOLO model
    model = YOLO(weights_path)
    
    # Configure model inference parameters
    model.conf = confidence_threshold
    
    # Apply model to target view
    target.apply_model(model, label_field=predictions_field)
    
    # Initialize return values
    wandb_url = None
    wandb_run_id = None
    total_predictions = 0
    final_artifact_name = None
    
    # Log predictions to W&B with lineage
    if log_to_wandb:
        # Collect sample IDs and label IDs for lineage
        sample_ids = target.values("id")
        all_label_ids = []
        for sample in target.iter_samples():
            pred = sample[predictions_field] if predictions_field in sample else None
            if pred:
                all_label_ids.extend(_collect_label_ids(pred))
        
        total_predictions = len(all_label_ids)
        
        # Calculate class distribution
        class_dist = _calculate_class_distribution(target, predictions_field)
        
        # Generate predictions artifact name (use provided name or auto-generate)
        if predictions_artifact_name:
            final_artifact_name = sanitize_for_artifact(predictions_artifact_name)
        else:
            safe_model = sanitize_for_artifact(model_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_artifact_name = f"{safe_model}_predictions_{timestamp}"
        
        # Create predictions artifact with lineage metadata
        predictions_artifact = wandb.Artifact(
            name=final_artifact_name,
            type="predictions",
            description=f"Predictions from {model_name} on {dataset.name}",
            metadata={
                # Model lineage
                "source_model_artifact": model_artifact,
                "model_name": model_name,
                "task_type": task_type,
                "confidence_threshold": confidence_threshold,
                
                # Dataset information
                "fiftyone_dataset_name": dataset.name,
                "fiftyone_dataset_size": len(dataset),
                "fiftyone_view_size": len(target),
                
                # Sample and label tracking for lineage
                "sample_ids": sample_ids,
                "label_ids": all_label_ids,
                "num_samples": len(sample_ids),
                "total_predictions": total_predictions,
                
                # Statistics
                "class_distribution": class_dist,
                "num_classes": len(class_dist),
                "predictions_field": predictions_field,
                
                # Timestamp
                "inference_timestamp": datetime.now().isoformat(),
            }
        )
        
        # Create W&B run and log with lineage
        wandb_run_id = run_name if run_name else f"inference_{uuid.uuid4().hex[:8]}"
        
        with wandb.init(project=project_name, id=wandb_run_id, name=run_name, entity=entity, reinit="finish_previous") as run:
            # Use the model artifact to create lineage (this links predictions -> model)
            run.use_artifact(model_artifact)
            
            # Log the predictions artifact
            logged_artifact = run.log_artifact(predictions_artifact)
            logged_artifact.wait()
            
            # Log run config
            run.config.update({
                "model_name": model_name,
                "model_artifact": model_artifact,
                "task_type": task_type,
                "predictions_field": predictions_field,
                "confidence_threshold": confidence_threshold,
                "samples_processed": len(target),
            })
        
        wandb_url = get_run_url(ctx, project_name, wandb_run_id)
    
    # Register inference run in FiftyOne
    run_config = dataset.init_run()
    run_config.method = "wandb_model_inference"
    run_config.model_artifact = model_artifact
    run_config.model_name = model_name
    run_config.task_type = task_type
    run_config.predictions_field = predictions_field
    run_config.confidence_threshold = confidence_threshold
    run_config.samples_processed = len(target)
    run_config.inference_timestamp = datetime.now().isoformat()
    run_config.artifact_metadata = artifact_metadata
    if final_artifact_name:
        run_config.predictions_artifact = f"{final_artifact_name}:latest"
    if wandb_url:
        run_config.wandb_url = wandb_url
    
    # Create run key
    safe_model = sanitize_for_run_key(model_name)
    run_key = f"inference_{safe_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset.register_run(run_key, run_config)
    
    return {
        "success": True,
        "model_artifact": model_artifact,
        "model_name": model_name,
        "task_type": task_type,
        "predictions_field": predictions_field,
        "samples_processed": len(target),
        "confidence_threshold": confidence_threshold,
        "total_predictions": total_predictions,
        "predictions_artifact": f"{final_artifact_name}:latest" if final_artifact_name else None,
        "run_name": wandb_run_id,
        "wandb_url": wandb_url,
    }


# ============================================================================
# OPERATOR
# ============================================================================

class ApplyYOLOModelFromRegistry(foo.Operator):
    """Apply a YOLO model from W&B Model Registry to a FiftyOne dataset."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="apply_yolo_model_from_registry",
            label="W&B: Apply YOLO Model from Registry",
            description="Select a YOLO model from the W&B Model Registry and run inference on your dataset",
            dynamic=True,
            icon="/assets/wandb.svg",
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )
    
    def __call__(
        self,
        sample_collection,
        project,
        model_artifact,
        predictions_field="predictions",
        confidence_threshold=0.25,
        log_to_wandb=False,
        run_name=None,
        predictions_artifact_name=None,
        delegate=False,
    ):
        """
        Programmatic interface for applying YOLO models from W&B registry.
        
        Args:
            sample_collection: FiftyOne dataset or view
            project: W&B project name containing the model
            model_artifact: Full artifact path (entity/project/name:version)
            predictions_field: Field to store predictions (default: "predictions")
            confidence_threshold: Minimum confidence for predictions (default: 0.25)
            log_to_wandb: Log predictions to W&B with lineage (default: False)
            run_name: Custom W&B run name (optional, auto-generated if not provided)
            predictions_artifact_name: Custom predictions artifact name (optional)
            delegate: Run in background (default: False)
            
        Returns:
            dict: Inference results including predictions artifact if logged
        """
        dataset = sample_collection._dataset
        view = sample_collection.view()
        
        ctx = create_mock_context(view, dataset, {
            "project": project,
            "model_artifact": model_artifact,
            "predictions_field": predictions_field,
            "confidence_threshold": confidence_threshold,
            "log_to_wandb": log_to_wandb,
            "run_name": run_name,
            "predictions_artifact_name": predictions_artifact_name,
        })
        
        return _apply_yolo_model(ctx)
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Check for ultralytics
        if not ULTRALYTICS_AVAILABLE:
            inputs.view("error", types.Error(
                label="Ultralytics Not Installed",
                description="Install ultralytics with: pip install ultralytics"
            ))
            return types.Property(inputs)
        
        # Check for W&B credentials
        if not prompt_for_missing_credentials(ctx, inputs):
            return types.Property(inputs)
        
        # Get credentials and API
        entity, api_key, project = get_credentials(ctx)
        
        # Pass credentials through as hidden params for delegated execution
        if entity and not ctx.params.get("wandb_entity"):
            inputs.str(
                "wandb_entity",
                default=entity,
                view=types.HiddenView(),
            )
        if api_key and not ctx.params.get("wandb_api_key"):
            inputs.str(
                "wandb_api_key",
                default=api_key,
                view=types.HiddenView(),
            )
        
        api = get_wandb_api(ctx)
        
        # ===== Target View Selection =====
        inputs.view("section_data", types.Header(label="Target Data", divider=True))
        inputs.view_target(ctx)
        
        target = ctx.target_view()
        inputs.view("view_info", types.Notice(
            label=f"Will run inference on {len(target)} samples",
            description=f"From dataset: {ctx.dataset.name}"
        ))
        
        # ===== W&B Project Selection =====
        inputs.view("section_wandb", types.Header(label="W&B Model Registry", divider=True))
        
        # Project selector
        projects = list(api.projects(entity=entity))
        project_choices = [types.Choice(label=p.name, value=p.name) for p in projects]
        
        inputs.enum(
            "project",
            [c.value for c in project_choices],
            label="W&B Project",
            description="Select the project containing your trained models",
            required=True,
            default=project,
            view=types.DropdownView()
        )
        
        # ===== Model Selection =====
        selected_project = ctx.params.get("project")
        model_artifacts = []
        
        if selected_project:
            model_artifacts = _get_model_artifacts(api, entity, selected_project)
        
        if model_artifacts:
            # Build choices with descriptive labels
            model_choices = []
            for artifact in model_artifacts:
                metadata = artifact.get("metadata", {})
                task_type = metadata.get("task_type", "unknown")
                model_name = metadata.get("model_name", "")
                
                # Format label: "artifact_name (task_type, base_model)"
                label_parts = [artifact["name"]]
                if task_type != "unknown":
                    label_parts.append(f"task: {task_type}")
                if model_name:
                    label_parts.append(f"base: {model_name}")
                
                label = f"{label_parts[0]}"
                if len(label_parts) > 1:
                    label += f" ({', '.join(label_parts[1:])})"
                
                model_choices.append(types.Choice(
                    label=label,
                    value=artifact["full_name"]
                ))
            
            inputs.enum(
                "model_artifact",
                [c.value for c in model_choices],
                label="Model",
                description="Select a model from the registry",
                required=True,
                view=types.DropdownView()
            )
            
            # Show model metadata if selected
            selected_artifact = ctx.params.get("model_artifact")
            if selected_artifact:
                # Find the selected artifact's metadata
                for artifact in model_artifacts:
                    if artifact["full_name"] == selected_artifact:
                        metadata = artifact.get("metadata", {})
                        
                        # Build info string
                        info_lines = []
                        if metadata.get("task_type"):
                            info_lines.append(f"Task: {metadata['task_type']}")
                        if metadata.get("model_name"):
                            info_lines.append(f"Base Model: {metadata['model_name']}")
                        if metadata.get("training_samples"):
                            info_lines.append(f"Training Samples: {metadata['training_samples']}")
                        if metadata.get("epochs"):
                            info_lines.append(f"Epochs: {metadata['epochs']}")
                        
                        # Add metrics if available
                        metric_keys = ["mAP50", "mAP50-95", "accuracy_top1", "precision", "recall"]
                        metrics = [f"{k}: {metadata[k]:.3f}" for k in metric_keys if k in metadata and metadata[k]]
                        if metrics:
                            info_lines.append(f"Metrics: {', '.join(metrics)}")
                        
                        if info_lines:
                            inputs.view("model_info", types.Notice(
                                label="Model Information",
                                description="\n".join(info_lines)
                            ))
                        break
        else:
            if selected_project:
                inputs.view("no_models", types.Warning(
                    label="No Models Found",
                    description=f"No model artifacts found in project '{selected_project}'. "
                                "Train a model first using 'W&B: Train YOLO Model'."
                ))
            else:
                inputs.view("select_project", types.Notice(
                    label="Select a Project",
                    description="Select a W&B project to see available models"
                ))
        
        # ===== Inference Configuration =====
        inputs.view("section_config", types.Header(label="Inference Configuration", divider=True))
        
        # Predictions field
        inputs.str(
            "predictions_field",
            label="Predictions Field",
            description="Field name to store model predictions",
            default="predictions",
            required=True
        )
        
        # Confidence threshold
        inputs.float(
            "confidence_threshold",
            label="Confidence Threshold",
            description="Minimum confidence score for predictions (0.0 - 1.0)",
            default=0.25,
        )
        
        # ===== W&B Logging Options =====
        inputs.view("section_logging", types.Header(label="W&B Logging", divider=True))
        
        inputs.bool(
            "log_to_wandb",
            label="Log predictions to W&B",
            description="Create a predictions artifact with lineage to the source model",
            default=False
        )
        
        # Only show these fields if log_to_wandb is enabled
        if ctx.params.get("log_to_wandb", False):
            inputs.str(
                "run_name",
                label="Run Name (optional)",
                description="Custom name for the W&B run (auto-generated if not provided)",
            )
            
            inputs.str(
                "predictions_artifact_name",
                label="Predictions Artifact Name (optional)",
                description="Custom name for the predictions artifact (auto-generated if not provided)",
            )
        
        # ===== Execution Options =====
        inputs.view("section_execution", types.Header(label="Execution Options", divider=True))
        
        inputs.bool(
            "delegate",
            label="Delegate execution",
            description="Run inference in background (recommended for large datasets)",
            default=False
        )
        
        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        """Delegate based on user checkbox selection."""
        return ctx.params.get("delegate", False)
    
    def execute(self, ctx):
        return _apply_yolo_model(ctx)
    
    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("model_artifact", label="Model Artifact")
        outputs.str("model_name", label="Model Name")
        outputs.str("task_type", label="Task Type")
        outputs.str("predictions_field", label="Predictions Field")
        outputs.int("samples_processed", label="Samples Processed")
        outputs.float("confidence_threshold", label="Confidence Threshold")
        outputs.int("total_predictions", label="Total Predictions")
        outputs.str("predictions_artifact", label="Predictions Artifact")
        outputs.str("run_name", label="W&B Run Name")
        outputs.str("wandb_url", label="W&B Run URL", view=types.LinkView())
        return types.Property(outputs)
