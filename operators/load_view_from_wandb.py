"""Load View from WandB operator.

This operator recreates a FiftyOne view from a WandB dataset artifact,
enabling reproducibility and experiment tracking.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import WANDB_AVAILABLE

try:
    import wandb
except ImportError:
    wandb = None


def _get_credentials(ctx):
    """Get W&B credentials from context.
    
    Returns:
        tuple: (entity, api_key) where either value may be None
    """
    entity = ctx.secrets.get("FIFTYONE_WANDB_ENTITY")
    api_key = ctx.secrets.get("FIFTYONE_WANDB_API_KEY")
    return entity, api_key


def _get_wandb_api(ctx):
    """Get authenticated W&B API client.
    
    Handles login once and returns API client.
    
    Args:
        ctx: Operator execution context
        
    Returns:
        wandb.Api: Authenticated API client
        
    Raises:
        ImportError: If wandb is not installed
        ValueError: If required credentials are missing
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb not installed. Run: pip install wandb")
    
    entity, api_key = _get_credentials(ctx)
    
    if not entity:
        raise ValueError("FIFTYONE_WANDB_ENTITY not set in secrets")
    
    # Login only if API key is provided
    if api_key:
        wandb.login(key=api_key, relogin=False)
    
    return wandb.Api()


def _validate_execution_params(ctx):
    """Validate parameters before execution.
    
    Args:
        ctx: Operator execution context
        
    Raises:
        ValueError: If required parameters are missing
    """
    if not ctx.params.get("project"):
        raise ValueError("Missing required parameter: project")
    
    if not ctx.params.get("artifact"):
        raise ValueError("Missing required parameter: artifact")


def _load_view_from_wandb(ctx):
    """Load view from WandB artifact"""
    
    # Validate parameters
    _validate_execution_params(ctx)
    
    # Get authenticated API client (handles validation and login)
    api = _get_wandb_api(ctx)
    entity, _ = _get_credentials(ctx)
    
    # Get parameters
    project = ctx.params.get("project")
    artifact_name = ctx.params.get("artifact")
    action = ctx.params.get("action", "apply_to_session")
    view_name = ctx.params.get("view_name")
    
    # Fetch artifact
    artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
    
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
        
        # Get credentials
        entity, api_key = _get_credentials(ctx)
        
        # Project selector
        if entity and api_key:
            # Get authenticated API (handles login once)
            try:
                api = _get_wandb_api(ctx)
            except (ImportError, ValueError) as e:
                inputs.view("error", types.Error(
                    label="Configuration Error",
                    description=str(e)
                ))
                return types.Property(inputs)
            
            projects = list(api.projects(entity=entity))
            project_choices = [types.Choice(label=p.name, value=p.name) for p in projects]
            
            inputs.enum(
                "project",
                [c.value for c in project_choices],
                label="W&B Project",
                required=True,
                default=ctx.secrets.get("FIFTYONE_WANDB_PROJECT"),
                view=types.DropdownView()
            )
        else:
            inputs.str("project", label="W&B Project", required=True)
            return types.Property(inputs)
        
        # Artifact selector (dataset artifacts only)  
        project = ctx.params.get("project")
        if project:
            # Reuse the API client from above (already authenticated)
            runs = api.runs(path=f"{entity}/{project}")
            artifact_names = set()
            
            for run in runs:
                for artifact in run.logged_artifacts():
                    if artifact.type == "dataset":
                        # Use the artifact name exactly as it appears in WandB
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
        if artifact_name:
            # Reuse the API client (already authenticated)
            full_artifact_path = f"{entity}/{project}/{artifact_name}"
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

