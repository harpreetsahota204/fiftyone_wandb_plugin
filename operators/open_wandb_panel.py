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
        # Use credentials to construct URL
        entity, _, project_name = get_credentials(ctx)
        
        # Try to load a recent run from the project (more likely to embed successfully)
        if entity and project_name:
            try:
                api = get_wandb_api(ctx)
                runs = list(api.runs(path=f"{entity}/{project_name}", per_page=1))
                if runs:
                    # Use the most recent run URL
                    url = runs[0].url
                else:
                    # No runs, fallback to project URL
                    url = get_project_url(ctx, project_name)
            except Exception as e:
                # If API call fails, fallback to project URL
                print(f"Could not fetch runs, using project URL: {e}")
                url = get_project_url(ctx, project_name)
        elif project_name:
            url = get_project_url(ctx, project_name)
        else:
            url = DEFAULT_WANDB_URL
        
        ctx.trigger(
            "@harpreetsahota/wandb/set_wandb_url",
            params=dict(url=url),
        )

