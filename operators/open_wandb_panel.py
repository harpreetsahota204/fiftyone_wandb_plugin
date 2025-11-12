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
)


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
        entity, _, project_name = get_credentials(ctx)
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

