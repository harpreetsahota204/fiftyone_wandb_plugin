"""Log FiftyOne View to W&B operator.

This operator logs a FiftyOne view (dataset or filtered subset) to W&B
as a dataset artifact during training, enabling dataset versioning and lineage tracking.
"""

import json
import os
import tempfile
from datetime import datetime

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


def _log_fiftyone_view_to_wandb(ctx):
    """Log FiftyOne view to WandB during training"""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Install it with: pip install wandb")
    
    view = ctx.view
    dataset = ctx.dataset
    run_id = ctx.params.get("run_id")
    project_name = ctx.params.get("project")
    artifact_name = ctx.params.get("artifact_name", None)
    
    # Auto-generate artifact name if not provided
    if not artifact_name:
        artifact_name = f"training_view_{run_id[:8]}"
    
    # Extract metadata
    metadata = extract_dataset_metadata(dataset, view)
    
    # Add view serialization for reproducibility
    is_subset = is_subset_view(view)
    if is_subset:
        metadata["view_serialization"] = serialize_view(view)
        metadata["is_subset_view"] = True
    else:
        metadata["is_subset_view"] = False
    
    metadata["is_training_view"] = True
    
    # Create WandB artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        description=f"Training view for run {run_id}",
        metadata=metadata
    )
    
    # Collect sample references (lightweight - no images)
    sample_refs = []
    target = view if is_subset else dataset
    
    for sample in target.iter_samples(progress=True):
        ref = {
            "id": sample.id,
            "filepath": sample.filepath,
            "tags": sample.tags if sample.tags else [],
        }
        
        # Add metadata if present
        if sample.metadata:
            ref["metadata"] = dict(sample.metadata)
        
        sample_refs.append(ref)
    
    # Write sample references to temp file and add to artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_refs, f, indent=2)
        temp_path = f.name
    
    try:
        artifact.add_file(temp_path, name="sample_references.json")
        
        # Get entity for WandB URLs
        entity = ctx.secret("FIFTYONE_WANDB_ENTITY")
        
        # Link artifact to run and log it
        with wandb.init(
            project=project_name,
            id=run_id,
            resume="must",
            entity=entity,
        ) as active_run:
            # Log the artifact (uploads it to WandB)
            active_run.log_artifact(artifact)
            
            # Also log key info to config for quick reference
            active_run.config.update({
                "fiftyone_view_artifact": f"{artifact_name}:latest",
                "fiftyone_dataset_name": dataset.name,
                "fiftyone_view_size": len(target),
                "fiftyone_is_subset": is_subset,
            })
        
        # Store training run info in FiftyOne using run system
        run_config = dataset.init_run()
        run_config.method = "wandb_training"
        run_config.view_serialization = serialize_view(view) if is_subset else None
        run_config.wandb_run_id = run_id
        run_config.wandb_run_url = f"https://wandb.ai/{entity}/{project_name}/runs/{run_id}"
        run_config.wandb_project = project_name
        run_config.dataset_artifact = f"{artifact_name}:latest"
        run_config.samples_used = len(target)
        run_config.status = "running"
        run_config.started_at = datetime.now().isoformat()
        
        # Register run in FiftyOne
        run_key = f"training_{run_id}"
        dataset.register_run(run_key, run_config)
        
        return {
            "success": True,
            "artifact_name": artifact_name,
            "run_key": run_key,
            "samples_logged": len(sample_refs),
            "wandb_url": f"https://wandb.ai/{entity}/{project_name}/runs/{run_id}",
        }
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


class LogFiftyOneViewToWandB(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="log_fiftyone_view_to_wandb",
            label="W&B: Log Training View",
            description="Log FiftyOne view to WandB as training dataset artifact",
            dynamic=True,
            icon="/assets/wandb.svg",
        )
    
    def __call__(
        self,
        sample_collection,
        project,
        run_id,
        artifact_name=None,
    ):
        """Programmatic interface for logging FiftyOne views to WandB"""
        dataset = sample_collection._dataset
        view = sample_collection.view()
        ctx = dict(view=view, dataset=dataset)
        params = dict(
            project=project,
            run_id=run_id,
            artifact_name=artifact_name,
        )
        return foo.execute_operator(self.uri, ctx, params=params)
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Check if in a view
        is_view = is_subset_view(ctx.view)
        if not is_view:
            inputs.view(
                "warning",
                types.Warning(
                    label="Full Dataset Selected",
                    description=f"You are logging the entire dataset ({len(ctx.dataset)} samples). "
                               "Consider creating a filtered view for training.",
                ),
            )
        else:
            inputs.view(
                "info",
                types.Notice(
                    label=f"Logging View with {len(ctx.view)} samples",
                    description=f"This view will be saved as a WandB dataset artifact. "
                               f"Total dataset has {len(ctx.dataset)} samples.",
                ),
            )
        
        # Project and run info
        inputs.str(
            "project",
            label="W&B Project",
            description="The W&B project name",
            required=True,
            default=ctx.secret("FIFTYONE_WANDB_PROJECT"),
        )
        
        inputs.str(
            "run_id",
            label="W&B Run ID",
            description="The ID of the current training run (from wandb.run.id)",
            required=True,
        )
        
        inputs.str(
            "artifact_name",
            label="Artifact Name (optional)",
            description="Name for the dataset artifact. Auto-generated if not provided.",
            required=False,
        )
        
        return types.Property(inputs)
    
    def execute(self, ctx):
        return _log_fiftyone_view_to_wandb(ctx)
    
    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("artifact_name", label="Artifact Name")
        outputs.str("run_key", label="FiftyOne Run Key")
        outputs.int("samples_logged", label="Samples Logged")
        outputs.str("wandb_url", label="WandB Run URL")
        return types.Property(outputs)

