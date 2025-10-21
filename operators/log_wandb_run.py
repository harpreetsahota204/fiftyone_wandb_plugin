"""Log W&B Run operator.

This operator logs a W&B run to a FiftyOne dataset, establishing
the connection between FiftyOne and W&B experiments.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import (
    WANDB_AVAILABLE,
    get_wandb_api,
    get_wandb_run,
    get_gt_field,
    is_subset_view,
    serialize_view,
    connect_predictions_to_run,
    connect_dataset_to_project_if_necessary,
    add_fiftyone_run_for_wandb_run,
)


def log_wandb_run(ctx):
    """Log W&B run to FiftyOne dataset"""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Install it with: pip install wandb")
    
    api = get_wandb_api(ctx)
    dataset = ctx.dataset
    view = ctx.view
    predictions_field = ctx.params.get("predictions_field", None)
    gt_field = ctx.params.get("gt_field", None)
    project_name = ctx.params.get("project", None)
    run_id = ctx.params.get("run_id", None)
    run_name = ctx.params.get("run_name", None)
    
    # Get the run
    run = get_wandb_run(ctx, project_name, run_id=run_id, run_name=run_name)
    if run is None:
        raise ValueError(f"Could not find W&B run. Project: {project_name}, "
                        f"Run ID: {run_id}, Run name: {run_name}")
    
    # Ensure project exists in FiftyOne
    connect_dataset_to_project_if_necessary(ctx, dataset, project_name)
    
    add_run_kwargs = {}
    
    # Connect predictions field
    if (
        predictions_field is not None
        and predictions_field in dataset.get_field_schema()
    ):
        connect_predictions_to_run(
            ctx,
            dataset,
            predictions_field,
            project_name,
            run,
        )
        add_run_kwargs["predictions_field"] = predictions_field
    
    if gt_field is not None and gt_field in dataset.get_field_schema():
        add_run_kwargs["gt_field"] = gt_field
    
    # Add view info if subset
    is_subset = is_subset_view(view)
    if is_subset:
        try:
            serial_view = serialize_view(view)
            run.tags = list(run.tags) + ["fiftyone_subset_view"]
            run.save()
        except Exception as e:
            print(f"Warning: Could not add subset view tag: {e}")
    
    # Add run to FiftyOne
    add_fiftyone_run_for_wandb_run(
        ctx, dataset, project_name, run, **add_run_kwargs
    )


class LogWandBRun(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="log_wandb_run",
            label="W&B: Log W&B run to the FiftyOne dataset",
            dynamic=True,
            unlisted=True,
        )
        _config.icon = "/assets/wandb.svg"
        return _config

    def __call__(
        self,
        sample_collection,
        project_name,
        run_id=None,
        run_name=None,
        predictions_field=None,
        gt_field=None,
    ):
        """Programmatic interface for logging W&B runs"""
        dataset = sample_collection._dataset
        view = sample_collection.view()
        ctx = dict(view=view, dataset=dataset)
        params = dict(
            project=project_name,
            run_id=run_id,
            run_name=run_name,
            predictions_field=predictions_field,
            gt_field=gt_field,
        )
        return foo.execute_operator(self.uri, ctx, params=params)

    def execute(self, ctx):
        log_wandb_run(ctx)

