"""Show W&B Run operator.

This operator displays W&B run data in the FiftyOne App, allowing users
to view run metrics and details without leaving FiftyOne.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import (
    DEFAULT_WANDB_URL,
    get_credentials,
    get_wandb_api,
    get_project_url,
    prompt_for_missing_credentials,
    WANDB_AVAILABLE,
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
        
        # Check for credentials and prompt if missing - all validation happens here
        if not prompt_for_missing_credentials(ctx, inputs):
            return types.Property(inputs)
        
        # Get credentials
        entity, _, project = get_credentials(ctx)
        
        # Fetch and show projects dropdown
        if entity and WANDB_AVAILABLE:
            # Get authenticated API (handles login once)
            try:
                api = get_wandb_api(ctx)
            except (ImportError, ValueError) as e:
                inputs.view("error", types.Error(
                    label="Configuration Error",
                    description=str(e)
                ))
                return types.Property(inputs)
            
            projects = list(api.projects(entity=entity))
            project_choices = [types.Choice(label=p.name, value=p.name) for p in projects]
            
            inputs.enum(
                "project_name",
                [c.value for c in project_choices],
                label="W&B Project",
                required=True,
                default=project,
                view=types.DropdownView()
            )
        else:
            inputs.str("project_name", label="W&B Project", required=True)
            return types.Property(inputs)
        
        # Get selected project
        project_name = ctx.params.get("project_name")
        if not project_name:
            return types.Property(inputs)
        
        # Fetch runs from W&B API (reuse API client from above)
        try:
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
        entity, _, project = get_credentials(ctx)
        entity = ctx.params.get("entity") or entity
        project_name = ctx.params.get("project_name") or project
        
        url = None
        
        if run_label and entity and project_name:
            # Fetch runs to find the URL matching the label
            try:
                api = get_wandb_api(ctx)
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
                url = get_project_url(ctx, project_name)
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

