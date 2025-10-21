"""Weights & Biases Experiment Tracking plugin.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_

"""

import json
import os
import tempfile
from datetime import datetime
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
        # Use environment variables to construct URL
        project_name = ctx.secret("FIFTYONE_WANDB_PROJECT")
        
        if project_name:
            url = _get_project_url(ctx, project_name)
        else:
            url = DEFAULT_WANDB_URL
        
        ctx.trigger(
            "@harpreetsahota/wandb/set_wandb_url",
            params=dict(url=url),
        )


class ShowWandBRun(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="show_wandb_run",
            label="Show W&B run",
            icon = "/assets/wandb.svg",
            dynamic=True,
            description=(
                "View the data and metrics for a W&B project/run"
                ", all in one place!"
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Get entity and project from environment
        entity = ctx.secret("FIFTYONE_WANDB_ENTITY")
        project_name = ctx.secret("FIFTYONE_WANDB_PROJECT")
        
        if not entity or not project_name:
            inputs.view(
                "warning",
                types.Warning(
                    label="Configuration Required",
                    description="Please set FIFTYONE_WANDB_ENTITY and FIFTYONE_WANDB_PROJECT "
                               "environment variables.",
                ),
            )
            # Allow manual input as fallback
            if not entity:
                inputs.str(
                    "entity",
                    label="W&B Entity",
                    description="Your W&B username or team name",
                    required=True,
                )
            if not project_name:
                inputs.str(
                    "project_name",
                    label="W&B Project",
                    description="The W&B project name",
                    required=True,
                )
            return types.Property(inputs)
        
        # Fetch runs from W&B API
        try:
            api = _get_wandb_api(ctx)
            runs = list(api.runs(path=f"{entity}/{project_name}", per_page=100))
            
            if len(runs) == 0:
                inputs.view(
                    "warning",
                    types.Warning(
                        label="No Runs Found",
                        description=f"No runs found for {entity}/{project_name}. "
                                   "The project page will be opened instead.",
                    ),
                )
                return types.Property(inputs)
            
            # Create dropdown of runs with summary metrics
            run_choices = types.DropdownView()
            for run in runs:
                # Simple format: run name and raw summary_metrics
                label = f"{run.name}"
                
                # Add summary metrics as raw string
                try:
                    summary_metrics = run.summary_metrics
                    if summary_metrics:
                        # Convert to string and show it
                        summary_str = str(summary_metrics)
                        # Truncate if too long (250 chars to match report description)
                        if len(summary_str) > 250:
                            summary_str = summary_str[:250] + "..."
                        label += f" | {summary_str}"
                except Exception as e:
                    # If we can't get summary, just use run name
                    pass
                
                run_choices.add_choice(label, label=label)
            
            inputs.enum(
                "run_label",
                run_choices.values(),
                label="Select Run",
                description="Choose a W&B run to view",
                required=False,
                view=types.DropdownView(),
            )
            
        except Exception as e:
            inputs.view(
                "error",
                types.Error(
                    label="Error Loading Runs",
                    description=f"Failed to fetch runs: {str(e)}",
                ),
            )
        
        return types.Property(inputs)

    def execute(self, ctx):
        # Get selected run label
        run_label = ctx.params.get("run_label", None)
        entity = ctx.params.get("entity") or ctx.secret("FIFTYONE_WANDB_ENTITY")
        project_name = ctx.params.get("project_name") or ctx.secret("FIFTYONE_WANDB_PROJECT")
        
        url = None
        
        if run_label and entity and project_name:
            # Fetch runs to find the URL matching the label
            try:
                api = _get_wandb_api(ctx)
                runs = list(api.runs(path=f"{entity}/{project_name}", per_page=100))
                
                # Find the run that matches the label (reconstruct same label format)
                for run in runs:
                    # Reconstruct simple label: run name + summary_metrics
                    label = f"{run.name}"
                    
                    try:
                        summary_metrics = run.summary_metrics
                        if summary_metrics:
                            summary_str = str(summary_metrics)
                            if len(summary_str) > 250:
                                summary_str = summary_str[:250] + "..."
                            label += f" | {summary_str}"
                    except:
                        pass
                    
                    if label == run_label:
                        url = run.url
                        break
            except Exception as e:
                print(f"Error fetching runs: {e}")
        
        # Fallback: open project page if no run selected or found
        if not url:
            if entity and project_name:
                url = _get_project_url(ctx, project_name)
            else:
                url = DEFAULT_WANDB_URL
        
        # Embed W&B run in iframe (runs CAN be embedded unlike main dashboard)
        ctx.trigger(
            "@harpreetsahota/wandb/embed_report",
            params=dict(url=url),
        )
        
        # Open the panel
        ctx.trigger(
            "open_panel",
            params=dict(name="WandBPanel", layout="horizontal", isActive=True),
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


class ShowWandBReport(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="show_wandb_report",
            label="Show W&B Report",
            icon = "/assets/wandb.svg",
            dynamic=True,
            description=(
                "View a W&B report embedded in FiftyOne. "
                "Reports can be embedded unlike the main dashboard."
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Get entity and project from environment or user input
        entity = ctx.secret("FIFTYONE_WANDB_ENTITY")
        project_name = ctx.secret("FIFTYONE_WANDB_PROJECT")
        
        if not entity or not project_name:
            inputs.view(
                "warning",
                types.Warning(
                    label="Configuration Required",
                    description="Please set FIFTYONE_WANDB_ENTITY and FIFTYONE_WANDB_PROJECT "
                               "environment variables to list reports.",
                ),
            )
            # Allow manual input
            if not entity:
                inputs.str(
                    "entity",
                    label="W&B Entity",
                    description="Your W&B username or team name",
                    required=True,
                )
            if not project_name:
                inputs.str(
                    "project_name",
                    label="W&B Project",
                    description="The W&B project name",
                    required=True,
                )
            return types.Property(inputs)
        
        # Fetch available reports
        try:
            api = _get_wandb_api(ctx)
            reports = list(api.reports(path=f"{entity}/{project_name}", per_page=100))
            
            if len(reports) == 0:
                inputs.view(
                    "warning",
                    types.Warning(
                        label="No Reports Found",
                        description=f"No reports found for {entity}/{project_name}. "
                                   "Create a report in W&B first.",
                    ),
                )
                return types.Property(inputs)
            
            # Create dropdown of reports with display names
            report_choices = types.DropdownView()
            
            for report in reports:
                # Build display label with name and description
                display_name = report.display_name or report.name
                if report.description:
                    # Truncate description if too long
                    desc = report.description[:250] + "..." if len(report.description) > 250 else report.description
                    label = f"{display_name} - {desc}"
                else:
                    label = f"{display_name} - No description provided"
                
                # Use label as the choice value (what user sees)
                report_choices.add_choice(label, label=label)
            
            inputs.enum(
                "report_label",
                report_choices.values(),
                label="Select Report",
                description="Choose a W&B report to embed",
                required=True,
                view=types.DropdownView(),
            )
            
        except Exception as e:
            inputs.view(
                "error",
                types.Error(
                    label="Error Loading Reports",
                    description=f"Failed to fetch reports: {str(e)}",
                ),
            )
        
        return types.Property(inputs)

    def execute(self, ctx):
        # Get selected report label
        report_label = ctx.params.get("report_label", None)
        report_url = None
        
        if report_label:
            # Fetch reports again to find the URL matching the label
            entity = ctx.secret("FIFTYONE_WANDB_ENTITY")
            project_name = ctx.secret("FIFTYONE_WANDB_PROJECT")
            
            if entity and project_name:
                try:
                    api = _get_wandb_api(ctx)
                    reports = list(api.reports(path=f"{entity}/{project_name}", per_page=100))
                    
                    # Find the report that matches the label
                    for report in reports:
                        display_name = report.display_name or report.name
                        if report.description:
                            desc = report.description[:250] + "..." if len(report.description) > 250 else report.description
                            label = f"{display_name} - {desc}"
                        else:
                            label = f"{display_name} - No description provided"
                        
                        if label == report_label:
                            report_url = report.url
                            break
                except Exception as e:
                    print(f"Error fetching reports: {e}")
        
        # Fallback if no report URL found
        if not report_url:
            entity = ctx.params.get("entity") or ctx.secret("FIFTYONE_WANDB_ENTITY")
            project_name = ctx.params.get("project_name") or ctx.secret("FIFTYONE_WANDB_PROJECT")
            
            if entity and project_name:
                # Open the reports page for the project
                report_url = f"{DEFAULT_WANDB_URL}/{entity}/{project_name}/reports"
            else:
                report_url = DEFAULT_WANDB_URL
        
        # Trigger the panel to embed the report
        ctx.trigger(
            "@harpreetsahota/wandb/embed_report",
            params=dict(url=report_url),
        )
        
        # Open the panel
        ctx.trigger(
            "open_panel",
            params=dict(name="WandBPanel", layout="horizontal", isActive=True),
        )


def _extract_dataset_metadata(dataset, view=None):
    """Extract metadata from FiftyOne dataset/view for WandB logging"""
    import fiftyone as fo
    
    target = view if view is not None else dataset
    
    metadata = {
        # Basic info
        "fiftyone_dataset_name": dataset.name,
        "fiftyone_dataset_size": len(dataset),
        "fiftyone_view_size": len(target),
        "fiftyone_version": fo.__version__,
        "fiftyone_media_type": dataset.media_type,
        
        # Timestamps
        "dataset_created_at": dataset.created_at.isoformat() if dataset.created_at else None,
        "dataset_last_modified_at": dataset.last_modified_at.isoformat() if dataset.last_modified_at else None,
        
        # Custom info
        "dataset_info": dict(dataset.info) if dataset.info else {},
    }
    
    # Add tag-based splits (check all common variations)
    all_tags = target.distinct("tags")
    for split in ["train", "val", "test", "validation", "training", "testing"]:
        if split in all_tags:
            metadata[f"{split}_samples"] = len(target.match_tags(split))
    
    # Add default classes if present
    if dataset.default_classes:
        metadata["default_classes"] = dataset.default_classes
    
    # Add mask targets if present (for segmentation)
    if hasattr(dataset, 'mask_targets') and dataset.mask_targets:
        metadata["mask_targets"] = dict(dataset.mask_targets)
    
    # Special handling for video datasets
    if dataset.media_type == "video":
        frame_schema = dataset.get_frame_field_schema()
        metadata["has_frame_fields"] = bool(frame_schema)
        if frame_schema:
            metadata["frame_fields"] = list(frame_schema.keys())
    
    return metadata


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
    metadata = _extract_dataset_metadata(dataset, view)
    
    # Add view serialization for reproducibility
    is_subset = _is_subset_view(view)
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
            # Use the artifact (links it to this run)
            active_run.use_artifact(artifact)
            
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
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Check if in a view
        is_view = _is_subset_view(ctx.view)
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


def register(p):
    """Register all operators"""
    p.register(OpenWandBPanel)
    p.register(GetWandBRunInfo)
    p.register(LogWandBRun)
    p.register(ShowWandBRun)
    p.register(ShowWandBReport)
    # Phase 1: Dataset Versioning & Lineage
    p.register(LogFiftyOneViewToWandB)
