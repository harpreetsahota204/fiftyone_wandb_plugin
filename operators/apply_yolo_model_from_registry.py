"""Apply YOLO Model from W&B Registry operator.

This operator allows users to select a YOLO model from the Weights & Biases
Model Registry and apply it to their FiftyOne dataset for inference.
"""

import os
from datetime import datetime

import fiftyone.operators as foo
import fiftyone.operators.types as types
import wandb

from ..wandb_helpers import (
    create_mock_context,
    ensure_wandb_login,
    get_credentials,
    get_wandb_api,
    prompt_for_missing_credentials,
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
            delegate: Run in background (default: False)
            
        Returns:
            dict: Inference results
        """
        dataset = sample_collection._dataset
        view = sample_collection.view()
        
        ctx = create_mock_context(view, dataset, {
            "project": project,
            "model_artifact": model_artifact,
            "predictions_field": predictions_field,
            "confidence_threshold": confidence_threshold,
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
            required=True
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
        """Delegate based on user checkbox or large dataset."""
        if ctx.params.get("delegate", False):
            return True
        # Auto-delegate for large datasets
        return len(ctx.target_view()) > 1000
    
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
        return types.Property(outputs)
