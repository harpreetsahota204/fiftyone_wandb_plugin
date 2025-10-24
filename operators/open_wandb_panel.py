"""Open W&B Panel operator.

This operator opens the W&B panel in the FiftyOne App.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import DEFAULT_WANDB_URL, get_project_url


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
        import os
        project_name = ctx.secrets.get("FIFTYONE_WANDB_PROJECT") or os.getenv("FIFTYONE_WANDB_PROJECT")
        
        if project_name:
            url = get_project_url(ctx, project_name)
        else:
            url = DEFAULT_WANDB_URL
        
        ctx.trigger(
            "@harpreetsahota/wandb/set_wandb_url",
            params=dict(url=url),
        )

