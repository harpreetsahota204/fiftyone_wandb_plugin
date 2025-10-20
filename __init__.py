"""Weights & Biases Experiment Tracking plugin.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_

"""

import json
import os
from bson import json_util

import fiftyone.operators as foo
import fiftyone.operators.types as types

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

DEFAULT_WANDB_URL = "https://wandb.ai"


def _get_wandb_config(ctx):
    """Get W&B configuration from secrets with fallbacks"""
    config = {
        "api_key": ctx.secret("FIFTYONE_WANDB_API_KEY"),
        "entity": ctx.secret("FIFTYONE_WANDB_ENTITY"),
        "project": ctx.secret("FIFTYONE_WANDB_PROJECT"),
    }
    return config


def _ensure_wandb_login(ctx):
    """Ensure W&B is logged in"""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Install it with: pip install wandb")
    
    api_key = ctx.secret("FIFTYONE_WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    # If no API key, wandb will use cached login or prompt


def _get_wandb_api(ctx):
    """Initialize W&B API client"""
    _ensure_wandb_login(ctx)
    api = wandb.Api()
    return api


def _get_project_path(ctx, project_name):
    """Get W&B project path in entity/project format"""
    config = _get_wandb_config(ctx)
    entity = config["entity"]
    if not entity:
        # Try to get from API
        api = _get_wandb_api(ctx)
        entity = api.default_entity
    return f"{entity}/{project_name}"


def _get_wandb_run(ctx, project_name, run_id=None, run_name=None):
    """Get W&B run by ID or name"""
    api = _get_wandb_api(ctx)
    project_path = _get_project_path(ctx, project_name)
    
    if run_id:
        # Get by ID
        return api.run(f"{project_path}/{run_id}")
    elif run_name:
        # Search by name
        runs = api.runs(project_path, filters={"display_name": run_name})
        if runs:
            return runs[0]
    
    return None


def _get_project_url(ctx, project_name):
    """Construct W&B project URL"""
    config = _get_wandb_config(ctx)
    entity = config["entity"]
    if not entity:
        api = _get_wandb_api(ctx)
        entity = api.default_entity
    return f"{DEFAULT_WANDB_URL}/{entity}/{project_name}"


def _get_run_url(ctx, project_name, run_id):
    """Construct W&B run URL"""
    project_url = _get_project_url(ctx, project_name)
    return f"{project_url}/runs/{run_id}"


def _format_run_name(run_name):
    """Format run name for FiftyOne (replace hyphens with underscores)"""
    return run_name.replace("-", "_")


def serialize_view(view):
    """Serialize a FiftyOne view"""
    return json.loads(json_util.dumps(view._serialize()))


def _get_gt_field(ctx, dataset):
    """Get ground truth field name"""
    if "gt_field" in ctx.params and ctx.params["gt_field"] is not None:
        return ctx.params["gt_field"]
    elif "ground_truth" in dataset.get_field_schema():
        return "ground_truth"
    else:
        return None


def _connect_predictions_to_run(
    ctx, dataset, predictions_field, project_name, run
):
    """Link predictions field to W&B run"""
    # Add run info to predictions field
    field = dataset.get_field(predictions_field)
    field.info = {
        "wandb_project": project_name,
        "wandb_run_name": run.name,
        "wandb_run_id": run.id,
        "wandb_url": run.url,
    }
    field.save()

    # Add tags to W&B run
    try:
        run.tags = list(run.tags) + ["fiftyone", f"predictions:{predictions_field}"]
        run.save()
    except Exception as e:
        print(f"Warning: Could not add tags to W&B run: {e}")

    # Add ground truth field tag
    gt_field = _get_gt_field(ctx, dataset)
    if gt_field is not None:
        try:
            run.tags = list(run.tags) + [f"ground_truth:{gt_field}"]
            run.save()
        except Exception as e:
            print(f"Warning: Could not add ground truth tag to W&B run: {e}")


def _initialize_fiftyone_run_for_wandb_project(
    ctx, dataset, project_name
):
    """Initialize FiftyOne run for W&B project"""
    config = dataset.init_run()
    wandb_config = _get_wandb_config(ctx)
    
    config.method = "wandb_project"
    config.entity = wandb_config["entity"]
    config.project_name = project_name
    config.project_url = _get_project_url(ctx, project_name)
    config.runs = []  # List of associated run IDs
    
    dataset.register_run(project_name, config)


def _add_fiftyone_run_for_wandb_run(
    ctx, dataset, project_name, run, **kwargs
):
    """Add W&B run to FiftyOne dataset"""
    config = dataset.init_run()
    wandb_config = _get_wandb_config(ctx)
    
    config.method = "wandb_run"
    config.run_name = run.name
    config.run_id = run.id
    config.entity = wandb_config["entity"]
    config.project = project_name
    config.state = run.state  # running, finished, crashed, etc.
    config.created_at = run.created_at
    config.url = run.url
    config.config_params = dict(run.config) if run.config else {}
    config.summary_metrics = dict(run.summary) if run.summary else {}
    config.tags = list(run.tags) if run.tags else []
    config.notes = run.notes if hasattr(run, 'notes') else ""
    
    if "predictions_field" in kwargs:
        config.predictions_field = kwargs["predictions_field"]
    if "gt_field" in kwargs:
        config.gt_field = kwargs["gt_field"]
    
    fmt_run_name = _format_run_name(run.name)
    dataset.register_run(fmt_run_name, config)
    
    # Add run to project's run list
    try:
        project_run_info = dataset.get_run_info(project_name)
        project_run_info.config.runs.append(run.name)
        dataset.update_run_config(project_name, project_run_info.config)
    except Exception as e:
        print(f"Warning: Could not update project run list: {e}")


def _is_subset_view(sample_collection):
    """Check if the sample collection is the entire dataset or a view"""
    return sample_collection.view() != sample_collection._dataset.view()


def _connect_dataset_to_project_if_necessary(
    ctx, dataset, project_name
):
    """Initialize FiftyOne project if it doesn't exist"""
    if project_name not in dataset.list_runs():
        _initialize_fiftyone_run_for_wandb_project(
            ctx, dataset, project_name
        )


def log_wandb_run(ctx):
    """Log W&B run to FiftyOne dataset"""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Install it with: pip install wandb")
    
    api = _get_wandb_api(ctx)
    dataset = ctx.dataset
    view = ctx.view
    predictions_field = ctx.params.get("predictions_field", None)
    gt_field = ctx.params.get("gt_field", None)
    project_name = ctx.params.get("project", None)
    run_id = ctx.params.get("run_id", None)
    run_name = ctx.params.get("run_name", None)
    
    # Get the run
    run = _get_wandb_run(ctx, project_name, run_id=run_id, run_name=run_name)
    if run is None:
        raise ValueError(f"Could not find W&B run. Project: {project_name}, "
                        f"Run ID: {run_id}, Run name: {run_name}")
    
    # Ensure project exists in FiftyOne
    _connect_dataset_to_project_if_necessary(ctx, dataset, project_name)
    
    add_run_kwargs = {}
    
    # Connect predictions field
    if (
        predictions_field is not None
        and predictions_field in dataset.get_field_schema()
    ):
        _connect_predictions_to_run(
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
    is_subset = _is_subset_view(view)
    if is_subset:
        try:
            serial_view = serialize_view(view)
            run.tags = list(run.tags) + ["fiftyone_subset_view"]
            run.save()
        except Exception as e:
            print(f"Warning: Could not add subset view tag: {e}")
    
    # Add run to FiftyOne
    _add_fiftyone_run_for_wandb_run(
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


def get_candidate_project_names(ctx):
    """Get list of W&B projects stored in FiftyOne"""
    project_names = [
        r
        for r in ctx.dataset.list_runs()
        if ctx.dataset.get_run_info(r).config.method == "wandb_project"
    ]
    return project_names


def get_candidate_run_names(ctx, project_name):
    """Get list of runs for a W&B project"""
    try:
        project_info = ctx.dataset.get_run_info(project_name)
        project_runs = project_info.config.runs
        return project_runs
    except Exception:
        return []


class OpenWandBPanel(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="open_wandb_panel",
            label="Open W&B Panel",
            unlisted=False,
        )
        _config.icon = "/assets/wandb.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Open W&B Panel",
                prompt=False,
                icon="/assets/wandb.svg",
            ),
        )

    def execute(self, ctx):
        # Open default W&B dashboard in new tab
        ctx.trigger(
            "@harpreetsahota/wandb/set_wandb_url",
            params=dict(url="https://wandb.ai"),
        )


class ShowWandBRun(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="show_wandb_run",
            label="Show W&B run",
            dynamic=True,
            description=(
                "View the data and metrics for a W&B project/run"
                ", all in one place!"
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        projects = get_candidate_project_names(ctx)
        if len(projects) == 0:
            inputs.view(
                "warning",
                types.Warning(
                    label="No W&B projects",
                    description="W&B home page will be opened instead. "
                               "Log a W&B run to your dataset first.",
                ),
            )
            return types.Property(inputs)

        project_choices = types.DropdownView()
        for project in projects:
            project_choices.add_choice(project, label=project)
        inputs.enum(
            "project_name",
            project_choices.values(),
            label="Project name",
            description="The name of the W&B project to display",
            required=True,
            view=types.DropdownView(),
        )

        project_name = ctx.params.get("project_name", None)
        if project_name is not None:
            runs = get_candidate_run_names(ctx, project_name)
            if len(runs) > 0:
                run_choices = types.DropdownView()
                for run in runs:
                    run_choices.add_choice(run, label=run)
                inputs.enum(
                    "run_name",
                    run_choices.values(),
                    label="Run name",
                    description="The name of the W&B run to display",
                    required=False,
                    view=types.DropdownView(),
                )

        return types.Property(inputs)

    def execute(self, ctx):
        project_name = ctx.params.get("project_name", None)
        run_name = ctx.params.get("run_name", None)
        
        # Fallback to environment variable if no project selected
        if project_name is None:
            project_name = ctx.secret("FIFTYONE_WANDB_PROJECT")
        
        # Construct URL
        if project_name is None:
            url = DEFAULT_WANDB_URL
        elif run_name is None:
            url = _get_project_url(ctx, project_name)
        else:
            # Get run info to get run ID
            fmt_run_name = _format_run_name(run_name)
            try:
                run_info = ctx.dataset.get_run_info(fmt_run_name)
                run_id = run_info.config.run_id
                url = _get_run_url(ctx, project_name, run_id)
                
                # Update view to show relevant fields
                keep_fields = []
                if hasattr(run_info.config, "gt_field"):
                    keep_fields.append(run_info.config.gt_field)
                if hasattr(run_info.config, "predictions_field"):
                    keep_fields.append(run_info.config.predictions_field)
                
                if keep_fields:
                    view = ctx.dataset.select_fields(keep_fields)
                    ctx.trigger(
                        "set_view",
                        params=dict(view=serialize_view(view)),
                    )
            except Exception as e:
                print(f"Warning: Could not get run info: {e}")
                url = _get_project_url(ctx, project_name)

        # Open W&B URL in new tab
        ctx.trigger(
            "@harpreetsahota/wandb/set_wandb_url",
            params=dict(url=url),
        )


def _initialize_run_output():
    """Initialize output schema for run info"""
    outputs = types.Object()
    outputs.str("run_key", label="Run key")
    outputs.str("timestamp", label="Creation time")
    outputs.str("version", label="FiftyOne version")
    outputs.obj("config", label="Config", view=types.JSONView())
    return outputs


def _execute_run_info(ctx, run_key):
    """Get run information"""
    info = ctx.dataset.get_run_info(run_key)

    timestamp = info.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    version = info.version
    config = info.config.serialize()
    config = {k: v for k, v in config.items() if v is not None}

    return {
        "run_key": run_key,
        "timestamp": timestamp,
        "version": version,
        "config": config,
    }


class GetWandBRunInfo(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="get_wandb_run_info",
            label="W&B: get run info",
            dynamic=True,
        )
        _config.icon = "/assets/wandb.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="W&B: choose run",
            description="Get information about a W&B run",
        )

        dataset = ctx.dataset
        run_keys = [
            r
            for r in dataset.list_runs()
            if dataset.get_run_info(r).config.method == "wandb_run"
        ]

        if len(run_keys) == 0:
            inputs.view(
                "warning",
                types.Warning(
                    label="No W&B runs found",
                    description="Log a W&B run to your dataset first.",
                ),
            )
            return types.Property(inputs, view=form_view)

        run_choices = types.DropdownView()
        for run_key in run_keys:
            run_choices.add_choice(run_key, label=run_key)

        inputs.enum(
            "run_key",
            run_choices.values(),
            label="Run key",
            description="The run to retrieve information for",
            required=True,
            view=types.DropdownView(),
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        run_key = ctx.params.get("run_key", None)
        return _execute_run_info(ctx, run_key)

    def resolve_output(self, ctx):
        outputs = _initialize_run_output()
        view = types.View(label="W&B run info")
        return types.Property(outputs, view=view)


def register(p):
    """Register all operators"""
    p.register(OpenWandBPanel)
    p.register(GetWandBRunInfo)
    p.register(LogWandBRun)
    p.register(ShowWandBRun)
