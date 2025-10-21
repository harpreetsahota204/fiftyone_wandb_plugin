"""Show W&B Run operator.

This operator displays W&B run data in the FiftyOne App, allowing users
to view run metrics and details without leaving FiftyOne.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import (
    DEFAULT_WANDB_URL,
    get_wandb_api,
    get_project_url,
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
            api = get_wandb_api(ctx)
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

