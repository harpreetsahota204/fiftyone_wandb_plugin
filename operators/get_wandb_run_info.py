"""Get W&B Run Info operator.

This operator retrieves and displays detailed information about a W&B run
stored in the FiftyOne dataset.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types


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
        run_keys = []
        for r in dataset.list_runs():
            try:
                run_info = dataset.get_run_info(r)
                if hasattr(run_info.config, 'method') and run_info.config.method == "wandb_run":
                    run_keys.append(r)
            except Exception:
                # Skip runs that can't be deserialized
                continue

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

