"""Log W&B Run operator.

This operator logs a W&B run to a FiftyOne dataset, establishing
the connection between FiftyOne and W&B experiments.

Note: This is an unlisted operator intended for programmatic use only.
"""

import fiftyone.operators as foo

from ..wandb_helpers import (
    add_fiftyone_run_for_wandb_run,
    connect_dataset_to_project_if_necessary,
    connect_predictions_to_run,
    create_mock_context,
    ensure_wandb_login,
    get_wandb_run,
    is_subset_view,
)


def _log_wandb_run(ctx):
    """Log W&B run to FiftyOne dataset.
    
    Args:
        ctx: Execution context with dataset, view, and params
        
    Raises:
        ImportError: If wandb is not installed
        ValueError: If the specified W&B run cannot be found
    """
    # Ensure logged in (raises ImportError if wandb not installed)
    ensure_wandb_login(ctx)
    
    dataset = ctx.dataset
    view = ctx.view
    project_name = ctx.params.get("project")
    run_id = ctx.params.get("run_id")
    run_name = ctx.params.get("run_name")
    predictions_field = ctx.params.get("predictions_field")
    gt_field = ctx.params.get("gt_field")
    
    # Get the run from W&B
    run = get_wandb_run(ctx, project_name, run_id=run_id, run_name=run_name)
    if run is None:
        raise ValueError(
            f"Could not find W&B run. Project: {project_name}, "
            f"Run ID: {run_id}, Run name: {run_name}"
        )
    
    # Ensure project exists in FiftyOne
    connect_dataset_to_project_if_necessary(ctx, dataset, project_name)
    
    add_run_kwargs = {}
    
    # Connect predictions field if specified and exists
    if predictions_field and predictions_field in dataset.get_field_schema():
        connect_predictions_to_run(ctx, dataset, predictions_field, project_name, run)
        add_run_kwargs["predictions_field"] = predictions_field
    
    # Pass ground truth field if specified and exists
    if gt_field and gt_field in dataset.get_field_schema():
        add_run_kwargs["gt_field"] = gt_field
    
    # Tag run if using a subset view
    if is_subset_view(view):
        try:
            run.tags = list(run.tags) + ["fiftyone_subset_view"]
            run.save()
        except Exception as e:
            print(f"Warning: Could not add subset view tag: {e}")
    
    # Register run in FiftyOne dataset
    add_fiftyone_run_for_wandb_run(ctx, dataset, project_name, run, **add_run_kwargs)
    
    return {
        "success": True,
        "run_name": run.name,
        "run_id": run.id,
        "project": project_name,
    }


class LogWandBRun(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="log_wandb_run",
            label="W&B: Log W&B run to the FiftyOne dataset",
            description="Register an existing W&B run with a FiftyOne dataset",
            icon="/assets/wandb.svg",
            dynamic=True,
            unlisted=True,  # Programmatic use only
        )

    def __call__(
        self,
        sample_collection,
        project_name,
        run_id=None,
        run_name=None,
        predictions_field=None,
        gt_field=None,
    ):
        """
        Programmatic interface for logging W&B runs to FiftyOne.
        
        Args:
            sample_collection: FiftyOne dataset or view
            project_name: W&B project name (required)
            run_id: W&B run ID (optional, provide either run_id or run_name)
            run_name: W&B run name (optional, provide either run_id or run_name)
            predictions_field: Field containing predictions to link (optional)
            gt_field: Ground truth field name (optional)
            
        Returns:
            dict: Result with run_name, run_id, and project
        """
        dataset = sample_collection._dataset
        view = sample_collection.view()
        
        ctx = create_mock_context(view, dataset, {
            "project": project_name,
            "run_id": run_id,
            "run_name": run_name,
            "predictions_field": predictions_field,
            "gt_field": gt_field,
        })
        
        return _log_wandb_run(ctx)

    def execute(self, ctx):
        return _log_wandb_run(ctx)