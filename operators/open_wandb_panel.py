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
        
        # Try to get a run URL (embeddable) instead of project/homepage (not embeddable)
        if entity and project_name:
            try:
                api = get_wandb_api(ctx)
                runs = list(api.runs(path=f"{entity}/{project_name}", per_page=1))
                if runs:
                    url = runs[0].url
                    print(f"🚀 Loading W&B run: {url}")
                else:
                    print(f"⚠️ No runs found in {entity}/{project_name}")
                    print(f"💡 Create a run in W&B first, or the panel may not embed properly")
            except Exception as e:
                print(f"⚠️ Could not fetch runs: {e}")
                print(f"💡 Set FIFTYONE_WANDB_ENTITY and FIFTYONE_WANDB_API_KEY to enable embedding")
        else:
            print(f"⚠️ W&B credentials not configured")
            print(f"💡 Set FIFTYONE_WANDB_ENTITY and FIFTYONE_WANDB_API_KEY environment variables")
        
        # If no run URL found, inform user and don't open panel
        if not url:
            print(f"❌ Cannot open W&B panel: No embeddable run URL available")
            print(f"💡 Either configure credentials or use 'Show W&B Run' operator instead")
            return
        
        # Load the run URL in the panel
        ctx.trigger(
            "@harpreetsahota/wandb/embed_report",
            params=dict(url=url),
        )
        
        # Open the panel
        ctx.trigger(
            "open_panel",
            params=dict(name="WandBPanel", layout="horizontal", isActive=True),
        )

