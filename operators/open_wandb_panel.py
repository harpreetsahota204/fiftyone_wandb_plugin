"""Open W&B Panel operator.

This operator opens the W&B panel in the FiftyOne App.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import (
    DEFAULT_WANDB_URL,
    get_credentials,
    get_project_url,
    get_wandb_api,
    prompt_for_missing_credentials,
)


class OpenWandBPanel(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="open_wandb_panel",
            label="Open W&B Panel",
            unlisted=False,
            dynamic=True,
        )
        _config.icon = "/assets/wandb.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Open W&B Panel",
                prompt=True,
                icon="/assets/wandb.svg",
            ),
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Check for credentials and prompt if missing
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
        
        # Fetch projects from W&B
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
        
        # Add helpful notice about how it works
        inputs.view(
            "panel_info",
            types.Notice(
                label="💡 How This Works",
                description=(
                    "We'll open a W&B panel in FiftyOne:\n"
                    "1. We'll load a recent run from your project\n"
                    "2. Once loaded, navigate within W&B to view Reports, Artifacts, etc.\n"
                    "3. The panel stays open so you can explore your W&B workspace"
                )
            )
        )
        
        return types.Property(inputs)

    def execute(self, ctx):
        entity, _, _ = get_credentials(ctx)
        project_name = ctx.params.get("project_name")
        url = None
        
        if entity and project_name:
            try:
                api = get_wandb_api(ctx)
                
                # Get a recent run from the project to use as entry point
                runs = list(api.runs(path=f"{entity}/{project_name}", per_page=1))
                if runs:
                    url = runs[0].url
                else:
                    # No runs available, fallback to project overview
                    url = f"{DEFAULT_WANDB_URL}/{entity}/{project_name}"
                    
            except Exception as e:
                print(f"Error fetching runs: {e}")
                url = f"{DEFAULT_WANDB_URL}/{entity}/{project_name}"
        
        # Fallback URLs if something went wrong
        if not url:
            if entity and project_name:
                url = f"{DEFAULT_WANDB_URL}/{entity}/{project_name}"
            else:
                url = DEFAULT_WANDB_URL
        
        # Show what we're loading
        print(f"🚀 Loading W&B: {url}")
        print(f"💡 Once loaded, you can navigate to Reports, Artifacts, and other sections within W&B")
        
        # Load the URL in the panel
        ctx.trigger(
            "@harpreetsahota/wandb/embed_report",
            params=dict(url=url),
        )
        
        # Open the panel
        ctx.trigger(
            "open_panel",
            params=dict(name="WandBPanel", layout="horizontal", isActive=True),
        )

