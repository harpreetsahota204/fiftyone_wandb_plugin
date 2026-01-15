"""Train YOLO Model operator with W&B integration.

This operator trains Ultralytics YOLO models on FiftyOne views and logs
all training artifacts, metrics, and model weights to Weights & Biases
with proper artifact lineage.
"""

import glob
import os
import tempfile
from datetime import datetime

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import (
    create_mock_context,
    ensure_wandb_login,
    get_artifact_versions,
    get_credentials,
    get_run_url,
    get_wandb_api,
    prompt_for_missing_credentials,
    sanitize_for_artifact,
    sanitize_for_run_key,
)

try:
    import wandb
except ImportError:
    wandb = None

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

# Available YOLO models by task type
YOLO_MODELS = {
    "classification": [
        "yolo11n-cls.pt",
        "yolo11s-cls.pt",
        "yolo11m-cls.pt",
        "yolo11l-cls.pt",
        "yolo11x-cls.pt",
        "yolov8n-cls.pt",
        "yolov8s-cls.pt",
        "yolov8m-cls.pt",
        "yolov8l-cls.pt",
        "yolov8x-cls.pt",
    ],
    "detection": [
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo11x.pt",
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
        "yolov5nu.pt",
        "yolov5su.pt",
        "yolov5mu.pt",
        "yolov5lu.pt",
        "yolov5xu.pt",
    ],
    "segmentation": [
        "yolo11n-seg.pt",
        "yolo11s-seg.pt",
        "yolo11m-seg.pt",
        "yolo11l-seg.pt",
        "yolo11x-seg.pt",
        "yolov8n-seg.pt",
        "yolov8s-seg.pt",
        "yolov8m-seg.pt",
        "yolov8l-seg.pt",
        "yolov8x-seg.pt",
    ],
    "pose": [
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
        "yolo11m-pose.pt",
        "yolo11l-pose.pt",
        "yolo11x-pose.pt",
        "yolov8n-pose.pt",
        "yolov8s-pose.pt",
        "yolov8m-pose.pt",
        "yolov8l-pose.pt",
        "yolov8x-pose.pt",
    ],
}

# Default training directory
TRAIN_ROOT = "/tmp/fiftyone_yolo_training"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _ensure_directories(base_dir):
    """Create necessary directories."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _get_label_field_for_task(dataset, task_type):
    """Auto-detect appropriate label field for the task type."""
    schema = dataset.get_field_schema()
    
    if task_type == "classification":
        # Look for Classification fields
        for name, field in schema.items():
            if hasattr(field, 'document_type'):
                doc_type_name = field.document_type.__name__
                if doc_type_name == "Classification":
                    return name
        # Fallback to common names
        for name in ["ground_truth", "label", "classification", "class"]:
            if name in schema:
                return name
                
    elif task_type == "detection":
        # Look for Detections fields
        for name, field in schema.items():
            if hasattr(field, 'document_type'):
                doc_type_name = field.document_type.__name__
                if doc_type_name == "Detections":
                    return name
        # Fallback to common names
        for name in ["ground_truth", "detections", "predictions"]:
            if name in schema:
                return name
                
    elif task_type == "segmentation":
        # Look for Segmentation or instance segmentation fields
        for name, field in schema.items():
            if hasattr(field, 'document_type'):
                doc_type_name = field.document_type.__name__
                if doc_type_name in ["Segmentation", "Detections"]:
                    return name
        for name in ["ground_truth", "segmentation", "masks"]:
            if name in schema:
                return name
                
    elif task_type == "pose":
        # Look for Keypoints fields
        for name, field in schema.items():
            if hasattr(field, 'document_type'):
                doc_type_name = field.document_type.__name__
                if doc_type_name == "Keypoints":
                    return name
        for name in ["ground_truth", "keypoints", "pose"]:
            if name in schema:
                return name
    
    return "ground_truth"


def _export_for_training(view, task_type, label_field, export_dir, split="train"):
    """Export FiftyOne view to format suitable for YOLO training."""
    
    if task_type == "classification":
        # Export as directory tree for classification
        view.export(
            export_dir=os.path.join(export_dir, split),
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            label_field=label_field,
        )
    elif task_type == "detection":
        # Get classes for detection
        classes = view.distinct(f"{label_field}.detections.label")
        view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split,
        )
    elif task_type == "segmentation":
        # Export for instance segmentation (YOLO format)
        classes = view.distinct(f"{label_field}.detections.label")
        view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split,
        )
    elif task_type == "pose":
        # Pose estimation export (YOLO format with keypoints)
        classes = view.distinct(f"{label_field}.keypoints.label") if hasattr(view, label_field) else ["person"]
        view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split,
        )
    
    return export_dir


def _log_training_visualizations(run, output_dir):
    """Log training visualizations to W&B."""
    logged_images = []
    
    # Training batch images
    for batch_path in sorted(glob.glob(f"{output_dir}/train_batch*.jpg")):
        batch_name = os.path.basename(batch_path).replace(".jpg", "")
        wandb.log({f"training/{batch_name}": wandb.Image(batch_path)})
        logged_images.append(batch_name)
    
    # Validation batch images
    for val_path in sorted(glob.glob(f"{output_dir}/val_batch*.jpg")):
        val_name = os.path.basename(val_path).replace(".jpg", "")
        wandb.log({f"validation/{val_name}": wandb.Image(val_path)})
        logged_images.append(val_name)
    
    # Confusion matrices
    if os.path.exists(f"{output_dir}/confusion_matrix.png"):
        wandb.log({"evaluation/confusion_matrix": wandb.Image(f"{output_dir}/confusion_matrix.png")})
        logged_images.append("confusion_matrix")
    
    if os.path.exists(f"{output_dir}/confusion_matrix_normalized.png"):
        wandb.log({"evaluation/confusion_matrix_normalized": wandb.Image(f"{output_dir}/confusion_matrix_normalized.png")})
        logged_images.append("confusion_matrix_normalized")
    
    # Results summary
    if os.path.exists(f"{output_dir}/results.png"):
        wandb.log({"evaluation/results": wandb.Image(f"{output_dir}/results.png")})
        logged_images.append("results")
    
    # PR curves (detection/segmentation)
    for pr_path in glob.glob(f"{output_dir}/*curve*.png"):
        pr_name = os.path.basename(pr_path).replace(".png", "")
        wandb.log({f"evaluation/{pr_name}": wandb.Image(pr_path)})
        logged_images.append(pr_name)
    
    # Labels distribution
    if os.path.exists(f"{output_dir}/labels.jpg"):
        wandb.log({"data/labels_distribution": wandb.Image(f"{output_dir}/labels.jpg")})
        logged_images.append("labels_distribution")
    
    return logged_images


def _extract_metrics(results, task_type):
    """Extract relevant metrics from YOLO training results."""
    metrics = {}
    results_dict = results.results_dict if hasattr(results, 'results_dict') else {}
    
    if task_type == "classification":
        metrics = {
            "accuracy_top1": results_dict.get("metrics/accuracy_top1", 0),
            "accuracy_top5": results_dict.get("metrics/accuracy_top5", 0),
            "val_loss": results_dict.get("val/loss", 0),
        }
    elif task_type == "detection":
        metrics = {
            "mAP50": results_dict.get("metrics/mAP50(B)", 0),
            "mAP50-95": results_dict.get("metrics/mAP50-95(B)", 0),
            "precision": results_dict.get("metrics/precision(B)", 0),
            "recall": results_dict.get("metrics/recall(B)", 0),
            "val_box_loss": results_dict.get("val/box_loss", 0),
            "val_cls_loss": results_dict.get("val/cls_loss", 0),
        }
    elif task_type == "segmentation":
        metrics = {
            "mAP50_box": results_dict.get("metrics/mAP50(B)", 0),
            "mAP50-95_box": results_dict.get("metrics/mAP50-95(B)", 0),
            "mAP50_mask": results_dict.get("metrics/mAP50(M)", 0),
            "mAP50-95_mask": results_dict.get("metrics/mAP50-95(M)", 0),
            "val_box_loss": results_dict.get("val/box_loss", 0),
            "val_seg_loss": results_dict.get("val/seg_loss", 0),
        }
    elif task_type == "pose":
        metrics = {
            "mAP50": results_dict.get("metrics/mAP50(P)", 0),
            "mAP50-95": results_dict.get("metrics/mAP50-95(P)", 0),
            "val_pose_loss": results_dict.get("val/pose_loss", 0),
        }
    
    return metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def _train_yolo_model(ctx):
    """Train YOLO model on FiftyOne view with W&B logging."""
    
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics is not installed. Install with: pip install ultralytics")
    
    # Ensure W&B is logged in (important for delegated execution where secrets may not be available)
    ensure_wandb_login(ctx)
    
    # Get parameters
    entity, _, _ = get_credentials(ctx)
    
    view = ctx.target_view()
    dataset = ctx.dataset
    
    # Required params (UI enforces these)
    project_name = ctx.params["project"]
    model_name = ctx.params["model_name"]
    task_type = ctx.params["task_type"]
    label_field = ctx.params["label_field"]
    
    # Training hyperparameters
    epochs = ctx.params.get("epochs", 10)
    image_size = ctx.params.get("image_size", 640)
    batch_size = ctx.params.get("batch_size", 16)
    
    # Additional hyperparameters
    learning_rate = ctx.params.get("learning_rate", 0.01)
    optimizer = ctx.params.get("optimizer", "auto")
    patience = ctx.params.get("patience", 50)
    
    # Optional: validation split (tag name)
    val_split = ctx.params.get("val_split")
    val_view = dataset.match_tags(val_split) if val_split else None
    
    # Optional: resume existing W&B run
    run_id = ctx.params.get("run_id")
    if not run_id:
        run_id = f"yolo_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Optional: use existing dataset artifact for lineage
    dataset_artifact_name = ctx.params.get("dataset_artifact")
    
    # Generate artifact name for the model
    model_artifact_name = ctx.params.get("model_artifact_name")
    if not model_artifact_name:
        safe_model = sanitize_for_artifact(model_name.replace(".pt", ""))
        model_artifact_name = f"{safe_model}_{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        model_artifact_name = sanitize_for_artifact(model_artifact_name)
    
    # Setup export directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(TRAIN_ROOT, f"{dataset.name}_{timestamp}")
    _ensure_directories(export_dir)
    
    # ========== PHASE 1: Export Data ==========
    # Export training view
    _export_for_training(view, task_type, label_field, export_dir, split="train")
    
    # Export validation view if provided
    if val_view is not None and len(val_view) > 0:
        _export_for_training(val_view, task_type, label_field, export_dir, split="val")
    
    # Determine data path for YOLO
    if task_type == "classification":
        data_path = export_dir  # Directory with train/val subdirs
    else:
        data_path = os.path.join(export_dir, "dataset.yaml")
    
    # ========== PHASE 2: Initialize W&B ==========
    # Ensure clean W&B state
    if wandb.run is not None:
        wandb.finish()
    
    run = wandb.init(
        project=project_name,
        entity=entity,
        id=run_id,
        resume="allow",
        config={
            "model": model_name,
            "task_type": task_type,
            "epochs": epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "patience": patience,
            "fiftyone_dataset": dataset.name,
            "training_samples": len(view),
            "validation_samples": len(val_view) if val_view else 0,
            "label_field": label_field,
        }
    )
    
    # Declare dataset artifact as input (creates lineage)
    dataset_artifact = None
    if dataset_artifact_name:
        dataset_artifact = run.use_artifact(dataset_artifact_name)
    
    # ========== PHASE 3: Train Model ==========
    # Initialize YOLO model
    model = YOLO(model_name)
    
    # Setup output directory for YOLO
    yolo_output_dir = os.path.join(TRAIN_ROOT, "runs", run_id)
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        lr0=learning_rate,
        optimizer=optimizer,
        patience=patience,
        project=yolo_output_dir,
        name="train",
        exist_ok=True,
    )
    
    # Get actual output directory
    train_output_dir = str(results.save_dir)
    
    # ========== PHASE 4: Log Metrics ==========
    # Extract and log metrics
    metrics = _extract_metrics(results, task_type)
    wandb.log(metrics)
    
    # Log training visualizations
    logged_images = _log_training_visualizations(run, train_output_dir)
    
    # ========== PHASE 5: Log Model Artifact ==========
    weights_path = os.path.join(train_output_dir, "weights", "best.pt")
    
    model_artifact = None
    if os.path.exists(weights_path):
        model_artifact = wandb.Artifact(
            name=model_artifact_name,
            type="model",
            description=f"YOLO {task_type} model trained on {dataset.name}",
            metadata={
                "model_name": model_name,
                "task_type": task_type,
                "source_dataset": dataset.name,
                "source_dataset_artifact": dataset_artifact.name if dataset_artifact else None,
                "training_samples": len(view),
                "validation_samples": len(val_view) if val_view else 0,
                "epochs": epochs,
                "image_size": image_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "patience": patience,
                "label_field": label_field,
                **metrics,  # Include final metrics
            }
        )
        model_artifact.add_file(weights_path, name="best.pt")
        
        logged_artifact = run.log_artifact(model_artifact)
        logged_artifact.wait()
    
    # Finish W&B run
    wandb.finish()
    
    # ========== PHASE 6: Register in FiftyOne ==========
    wandb_url = get_run_url(ctx, project_name, run_id)
    
    # Register training run in FiftyOne
    run_config = dataset.init_run()
    run_config.method = "yolo_training"
    run_config.model_name = model_name
    run_config.task_type = task_type
    run_config.epochs = epochs
    run_config.learning_rate = learning_rate
    run_config.optimizer = optimizer
    run_config.patience = patience
    run_config.training_samples = len(view)
    run_config.wandb_project = project_name
    run_config.wandb_run_id = run_id
    run_config.wandb_url = wandb_url
    run_config.model_artifact = f"{model_artifact_name}:latest" if model_artifact else None
    run_config.weights_path = weights_path if os.path.exists(weights_path) else None
    run_config.metrics = metrics
    run_config.trained_at = datetime.now().isoformat()
    
    run_key = sanitize_for_run_key(f"yolo_{task_type}_{run_id}")
    dataset.register_run(run_key, run_config)
    
    return {
        "success": True,
        "model_name": model_name,
        "task_type": task_type,
        "epochs": epochs,
        "training_samples": len(view),
        "metrics": metrics,
        "model_artifact": f"{model_artifact_name}:latest" if model_artifact else None,
        "weights_path": weights_path if os.path.exists(weights_path) else None,
        "wandb_url": wandb_url,
        "logged_visualizations": logged_images,
    }


# ============================================================================
# OPERATOR
# ============================================================================

class TrainYOLOModel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="train_yolo_model",
            label="W&B: Train YOLO Model",
            description="Train an Ultralytics YOLO model on your view and log results to W&B",
            dynamic=True,
            icon="/assets/wandb.svg",
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )
    
    def __call__(
        self,
        sample_collection,
        model_name,
        task_type,
        project,
        label_field=None,
        epochs=10,
        image_size=640,
        batch_size=16,
        learning_rate=0.01,
        optimizer="auto",
        patience=50,
        val_split=None,
        run_id=None,
        dataset_artifact=None,
        model_artifact_name=None,
        delegate=False
    ):
        """
        Programmatic interface for training YOLO models with W&B logging.
        
        Args:
            sample_collection: FiftyOne dataset or view to train on
            model_name: YOLO model name (e.g., 'yolo11n-cls.pt', 'yolov8n.pt')
            task_type: One of 'classification', 'detection', 'segmentation', 'pose'
            project: W&B project name
            label_field: Field containing labels (auto-detected if not provided)
            epochs: Number of training epochs (default: 10)
            image_size: Image size for training (default: 640)
            batch_size: Batch size (default: 16)
            learning_rate: Initial learning rate (default: 0.01)
            optimizer: Optimizer - 'auto', 'SGD', 'Adam', 'AdamW', etc. (default: 'auto')
            patience: Early stopping patience in epochs (default: 50)
            val_split: Tag name for validation samples (e.g., "val")
            run_id: W&B run ID to resume (auto-generated if not provided)
            dataset_artifact: W&B dataset artifact name for lineage
            model_artifact_name: Custom name for output model artifact
            
        Returns:
            dict: Training results including metrics, artifact names, and W&B URL
        """
        dataset = sample_collection._dataset
        view = sample_collection.view()
        
        if label_field is None:
            label_field = _get_label_field_for_task(dataset, task_type)
        
        ctx = create_mock_context(view, dataset, {
            "project": project,
            "model_name": model_name,
            "task_type": task_type,
            "label_field": label_field,
            "epochs": epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "patience": patience,
            "val_split": val_split,
            "run_id": run_id,
            "dataset_artifact": dataset_artifact,
            "model_artifact_name": model_artifact_name,
        })
        
        return _train_yolo_model(ctx)
    
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
        
        # IMPORTANT: Pass credentials through as hidden params for delegated execution
        # When delegating, ctx.secrets won't be available, so we need to pass credentials
        # explicitly in params. These are hidden from the UI.
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
        
        try:
            api = get_wandb_api(ctx)
        except (ImportError, ValueError) as e:
            inputs.view("error", types.Error(
                label="Configuration Error",
                description=str(e)
            ))
            return types.Property(inputs)
        
        # ===== View Selection =====
        inputs.view("section_data", types.Header(label="Training Data", divider=True))
        inputs.view_target(ctx)
        
        # Show view info
        target = ctx.target_view()
        inputs.view("view_info", types.Notice(
            label=f"Training view: {len(target)} samples",
            description=f"From dataset: {ctx.dataset.name}"
        ))
        
        # ===== Task & Model Selection =====
        inputs.view("section_model", types.Header(label="Model Configuration", divider=True))
        
        # Task type selector
        task_group = types.RadioGroup()
        task_group.add_choice("classification", label="Classification")
        task_group.add_choice("detection", label="Object Detection")
        task_group.add_choice("segmentation", label="Instance Segmentation")
        task_group.add_choice("pose", label="Pose Estimation")
        
        inputs.enum(
            "task_type",
            task_group.values(),
            label="Task Type",
            default="detection",
            required=True,
            view=task_group
        )
        
        # Model selector based on task type
        task_type = ctx.params.get("task_type", "detection")
        available_models = YOLO_MODELS.get(task_type, YOLO_MODELS["detection"])
        model_choices = [types.Choice(label=m, value=m) for m in available_models]
        
        inputs.enum(
            "model_name",
            [c.value for c in model_choices],
            label="YOLO Model",
            description="Select model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
            required=True,
            default=available_models[0],
            view=types.DropdownView()
        )
        
        # Label field (auto-detect with option to override)
        detected_field = _get_label_field_for_task(ctx.dataset, task_type)
        schema = ctx.dataset.get_field_schema()
        label_fields = [f for f in schema.keys() if f not in ["id", "filepath", "metadata", "tags"]]
        
        if label_fields:
            field_choices = [types.Choice(label=f, value=f) for f in label_fields]
            inputs.enum(
                "label_field",
                [c.value for c in field_choices],
                label="Label Field",
                description="Field containing ground truth labels",
                required=True,
                default=detected_field if detected_field in label_fields else label_fields[0],
                view=types.AutocompleteView(choices=field_choices)
            )
        else:
            inputs.str("label_field", label="Label Field", required=True, default="ground_truth")
        
        # ===== Training Hyperparameters =====
        inputs.view("section_training", types.Header(label="Training Parameters", divider=True))
        
        inputs.int("epochs", label="Epochs", default=10, required=True,
                   description="Number of training epochs")
        inputs.int("image_size", label="Image Size", default=640, required=True,
                   description="Input image size (pixels)")
        inputs.int("batch_size", label="Batch Size", default=16, required=True,
                   description="Training batch size")
        
        # Additional hyperparameters
        inputs.float("learning_rate", label="Learning Rate", default=0.01, required=True,
                     description="Initial learning rate (lr0)")
        
        optimizer_choices = types.DropdownView()
        optimizer_choices.add_choice("auto", label="Auto (recommended)")
        optimizer_choices.add_choice("SGD", label="SGD")
        optimizer_choices.add_choice("Adam", label="Adam")
        optimizer_choices.add_choice("AdamW", label="AdamW")
        optimizer_choices.add_choice("NAdam", label="NAdam")
        optimizer_choices.add_choice("RAdam", label="RAdam")
        optimizer_choices.add_choice("RMSProp", label="RMSProp")
        
        inputs.enum(
            "optimizer",
            optimizer_choices.values(),
            label="Optimizer",
            default="auto",
            required=True,
            description="Optimizer for training",
            view=optimizer_choices
        )
        
        inputs.int(
            "patience", 
            label="Early Stopping Patience", 
            default=50, 
            required=True,
            description="Epochs to wait before early stopping (0 to disable)"
            )
        
        # Optional validation split
        all_tags = ctx.dataset.distinct("tags")
        if all_tags:
            val_tags = [t for t in all_tags if any(v in t.lower() for v in ["val", "test", "eval"])]
            if val_tags:
                tag_choices = [types.Choice(label=t, value=t) for t in val_tags]
                inputs.enum(
                    "val_split",
                    [c.value for c in tag_choices],
                    label="Validation Split Tag (optional)",
                    description="Tag identifying validation samples",
                    view=types.DropdownView()
                )
        
        # ===== W&B Configuration =====
        inputs.view("section_wandb", types.Header(label="W&B Configuration", divider=True))
        
        # Project selector
        projects = list(api.projects(entity=entity))
        project_choices = [types.Choice(label=p.name, value=p.name) for p in projects]
        
        inputs.enum(
            "project",
            [c.value for c in project_choices],
            label="W&B Project",
            required=True,
            default=project,
            view=types.DropdownView()
        )
        
        # Fetch runs and artifacts from selected project
        project_name = ctx.params.get("project")
        existing_runs = []
        artifact_names = set()
        
        if project_name:
            runs = list(api.runs(path=f"{entity}/{project_name}", per_page=50))
            existing_runs = runs
            
            # Use efficient artifact API - returns versioned names like "my_artifact:v0"
            versioned_names = get_artifact_versions(api, entity, project_name, type_name="dataset")
            for name in versioned_names:
                # Use full qualified name: entity/project/name:version
                full_name = f"{entity}/{project_name}/{name}"
                artifact_names.add(full_name)
        
        # Run ID selector - allows selecting existing run OR typing new ID
        if existing_runs:
            # Build choices with run ID as value and descriptive label
            run_choices = []
            for run in existing_runs:
                # Format: "run_id - run_name (state)"
                label = f"{run.id} - {run.name} ({run.state})"
                run_choices.append(types.Choice(label=label, value=run.id))
            
            inputs.str(
                "run_id",
                label="W&B Run ID (optional)",
                description="Select existing run to resume, type custom ID, or leave empty to auto-generate",
                view=types.AutocompleteView(choices=run_choices)
            )
        else:
            # No existing runs - just show text input
            inputs.str(
                "run_id",
                label="W&B Run ID (optional)",
                description="Leave empty to auto-generate, or enter custom ID for new run"
            )
        
        # Optional: link to existing dataset artifact
        if artifact_names:
            artifact_choices = [types.Choice(label=a, value=a) for a in sorted(artifact_names)]
            inputs.enum(
                "dataset_artifact",
                [c.value for c in artifact_choices],
                label="Link Dataset Artifact (optional)",
                description="Creates artifact lineage in W&B",
                view=types.AutocompleteView(choices=artifact_choices)
            )
        
        # Model artifact name
        inputs.str(
            "model_artifact_name",
            label="Model Artifact Name (optional)",
            description="Custom name for the output model artifact (auto-generated if empty)"
        )
        
        # ===== Execution Options =====
        inputs.view(
            "section_execution", 
            types.Header(
                label="Execution Options", 
                divider=True
                )
            )
        
        inputs.bool(
            "delegate",
            label="Delegate execution",
            description="Run training in background (recommended for longer training jobs)",
            default=False
        )
        
        # ===== Summary =====
        model_name = ctx.params.get("model_name", "")
        epochs = ctx.params.get("epochs", 10)
        lr = ctx.params.get("learning_rate", 0.01)
        opt = ctx.params.get("optimizer", "auto")
        
        inputs.view("summary", types.Notice(
            label="Ready to train",
            description=f"Will train {model_name} for {epochs} epochs on {len(target)} samples (lr={lr}, optimizer={opt})"
        ))
        
        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        """Delegate based on user checkbox selection."""
        return ctx.params.get("delegate", False)
    
    def execute(self, ctx):
        return _train_yolo_model(ctx)
    
    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("model_name", label="Model")
        outputs.str("task_type", label="Task Type")
        outputs.int("epochs", label="Epochs")
        outputs.int("training_samples", label="Training Samples")
        outputs.obj("metrics", label="Final Metrics")
        outputs.str("model_artifact", label="Model Artifact")
        outputs.str("weights_path", label="Weights Path")
        outputs.str("wandb_url", label="W&B Run URL", view=types.LinkView())
        return types.Property(outputs)
