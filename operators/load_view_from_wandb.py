"""Load View from WandB operator.

This operator recreates a FiftyOne view from a WandB dataset artifact,
enabling reproducibility and experiment tracking.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import (
    WANDB_AVAILABLE,
    get_credentials,
    get_wandb_api,
    prompt_for_missing_credentials,
)

try:
    import wandb
except ImportError:
    wandb = None


def _load_view_from_wandb(ctx):
    """Load view from WandB artifact"""
    
    # Get credentials and API client (handles login and validation)
    entity, _, _ = get_credentials(ctx)
    api = get_wandb_api(ctx)
    
    # Get parameters (UI enforces required=True, so these will be present)
    project_name = ctx.params["project"]
    artifact_name = ctx.params["artifact"]
    action = ctx.params.get("action", "apply_to_session")
    view_name = ctx.params.get("view_name")
    
    # Fetch artifact
    artifact = api.artifact(f"{entity}/{project_name}/{artifact_name}")
    
    # Get sample IDs from metadata
    sample_ids = artifact.metadata["sample_ids"]
    
    # Recreate view
    dataset = ctx.dataset
    view = dataset.select(sample_ids)
    
    # Apply to session
    if action in ["apply_to_session", "both"]:
        ctx.trigger("set_view", params={"view": view._serialize()})
    
    # Save as named view
    if action in ["save_as_view", "both"] and view_name:
        dataset.save_view(view_name, view)
    
    return {
        "success": True,
        "samples_loaded": len(view),
        "view_name": view_name if action != "apply_to_session" else None,
        "artifact_name": artifact_name,
    }


class LoadViewFromWandB(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="load_view_from_wandb",
            label="W&B: Load View from Artifact",
            description="Recreate a FiftyOne view from a WandB dataset artifact",
            dynamic=True,
            icon="/assets/wandb.svg",
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Prompt for credentials if missing - all validation happens here
        if not prompt_for_missing_credentials(ctx, inputs):
            return types.Property(inputs)
        
        # Get credentials and API
        entity, _, project = get_credentials(ctx)
        
        try:
            api = get_wandb_api(ctx)
        except (ImportError, ValueError) as e:
            inputs.view("error", types.Error(
                label="Configuration Error",
                description=str(e)
            ))
            return types.Property(inputs)
        
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
        
        # Artifact selector (dataset artifacts only)  
        project_name = ctx.params.get("project")
        if project_name:
            runs = api.runs(path=f"{entity}/{project_name}")
            artifact_names = set()
            
            for run in runs:
                for artifact in run.logged_artifacts():
                    if artifact.type == "dataset":
                        artifact_names.add(artifact.name)
            
            artifact_choices = [
                types.Choice(label=name, value=name)
                for name in sorted(artifact_names)
            ]
            
            if artifact_choices:
                inputs.enum(
                    "artifact",
                    [c.value for c in artifact_choices],
                    label="Dataset Artifact",
                    description="Select artifact containing saved view",
                    required=True,
                    view=types.DropdownView()
                )
            else:
                inputs.view("info", types.Notice(
                    label="No artifacts found",
                    description="No dataset artifacts found in this project"
                ))
                return types.Property(inputs)
        else:
            return types.Property(inputs)
        
        # Show artifact info
        artifact_name = ctx.params.get("artifact")
        if artifact_name and project_name:
            full_artifact_path = f"{entity}/{project_name}/{artifact_name}"
            artifact = api.artifact(full_artifact_path)
            num_samples = artifact.metadata.get("num_samples", 0)
            dataset_name = artifact.metadata.get("fiftyone_dataset_name", "unknown")
            
            inputs.view("artifact_info", types.Notice(
                label=f"Will load {num_samples} samples",
                description=f"From dataset: {dataset_name}"
            ))
        
        # Action selector
        action_group = types.RadioGroup()
        action_group.add_choice("apply_to_session", label="Apply to current session")
        action_group.add_choice("save_as_view", label="Save as named view")
        action_group.add_choice("both", label="Apply to session AND save")
        
        inputs.enum(
            "action",
            action_group.values(),
            label="Action",
            default="apply_to_session",
            view=action_group
        )
        
        # View name (if saving)
        if ctx.params.get("action") in ["save_as_view", "both"]:
            inputs.str(
                "view_name",
                label="View Name",
                description="Name for the saved view",
                required=True
            )
        
        return types.Property(inputs)
    
    def execute(self, ctx):
        return _load_view_from_wandb(ctx)
    
    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("samples_loaded", label="Samples Loaded")
        outputs.str("artifact_name", label="Artifact Name")
        
        view_name = ctx.params.get("view_name")
        if view_name:
            outputs.str("view_name", label="Saved View Name")
        
        return types.Property(outputs)

