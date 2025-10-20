# Weights & Biases Experiment Tracking Plugin for FiftyOne

Training models is hard, and bridging the divide between data and models is even harder.
Fortunately, the right tooling can make data-model co-development a whole lot easier.

This plugin integrates [FiftyOne](https://docs.voxel51.com/) with [Weights & Biases](https://wandb.ai/) to provide a seamless experience for tracking, visualizing, and comparing your datasets and models.

## What is FiftyOne?

[FiftyOne](https://docs.voxel51.com/) is an open-source tool for data exploration and debugging in computer vision. It provides a powerful Python API for working with datasets, and a web-based UI for visualizing and interacting with your data.

## What is Weights & Biases?

[Weights & Biases](https://wandb.ai/) is a leading ML platform for experiment tracking, dataset versioning, and model management. It provides tools for tracking experiments, visualizing metrics, collaborating with teams, and managing the complete machine learning lifecycle.

## What does this plugin do?

This plugin helps you to connect your W&B experiments (projects and runs) to your FiftyOne datasets for enhanced tracking, visualization, model comparison, and debugging!

You can use this plugin to:

- Connect your W&B projects and runs to your FiftyOne datasets
- Visualize the W&B dashboard right beside your FiftyOne dataset in the FiftyOne App
- Get helpful information about your W&B runs and projects in the FiftyOne App
- Link predictions and ground truth fields to specific W&B runs
- Track which datasets were used for which experiments

## Installation

First, install the dependencies:

```bash
pip install -U fiftyone wandb
```

Then, download the plugin:

```bash
fiftyone plugins download https://github.com/harpreetsahota204/fiftyone_wandb_plugin
```

## Setup

### 1. W&B Authentication

You need to authenticate with W&B. You have several options:

**Option A: Set environment variables (Recommended)**

```bash
export WANDB_API_KEY="your-api-key-here"
export WANDB_ENTITY="your-username-or-team"  # Optional
export WANDB_PROJECT="your-default-project"   # Optional
```

You can find your API key at [https://wandb.ai/authorize](https://wandb.ai/authorize).

**Option B: Log in via command line**

```bash
wandb login
```

This will cache your credentials locally.

### 2. Configure FiftyOne Secrets (Optional)

If you prefer to configure secrets through FiftyOne:

```python
import fiftyone as fo

# Set W&B secrets
fo.set_secret("WANDB_API_KEY", "your-api-key-here")
fo.set_secret("WANDB_ENTITY", "your-username-or-team")
fo.set_secret("WANDB_PROJECT", "your-default-project")
```

## Usage

### Basic Example

Here is a basic template for using the plugin:

```python
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz

# Load a dataset
dataset = foz.load_zoo_dataset("quickstart")

# Get the log W&B run operator
log_wandb_run = foo.get_operator("@harpreetsahota/wandb/log_wandb_run")
```

### Logging a W&B Run to FiftyOne

After you've run an experiment with W&B, connect it to your FiftyOne dataset:

```python
project_name = "your-wandb-project"
run_name = "your-run-name"  # Or use run_id
predictions_field = "predictions"  # Optional: field containing model predictions

# Log the run
log_wandb_run(
    dataset, 
    project_name=project_name,
    run_name=run_name,
    predictions_field=predictions_field
)
```

You can also use `run_id` instead of `run_name`:

```python
log_wandb_run(
    dataset,
    project_name="my-project",
    run_id="abc123xyz",  # W&B run ID
    predictions_field="predictions"
)
```

### Viewing W&B Runs in FiftyOne App

In the FiftyOne App, you can visualize your W&B runs using the `show_wandb_run` operator, which will open the W&B dashboard within the app (or update it if already open):

1. **Via Python:**
```python
show_wandb_run = foo.get_operator("@harpreetsahota/wandb/show_wandb_run")
show_wandb_run(dataset)
```

2. **Via FiftyOne App:**
- Click the W&B button in the samples grid
- Or use the operators menu: `W&B: Show W&B run`
- Select a project and optionally a specific run
- The W&B dashboard will open in an embedded panel

### Getting W&B Run Information

You can get detailed information about runs stored in your dataset:

```python
get_run_info = foo.get_operator("@harpreetsahota/wandb/get_wandb_run_info")
info = get_run_info(dataset)
```

This will show:
- Run configuration
- Summary metrics
- Tags
- URLs
- Timestamps

## Complete Workflow Example

Here's a complete example showing how to integrate W&B with FiftyOne:

```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo
import wandb

# 1. Load a dataset
dataset = foz.load_zoo_dataset("quickstart")

# 2. Train a model and log to W&B
run = wandb.init(
    project="fiftyone-integration-demo",
    config={
        "dataset": dataset.name,
        "dataset_size": len(dataset),
        "model": "resnet50",
    }
)

# Train your model...
# predictions = model.predict(dataset)

# Log metrics
run.log({"accuracy": 0.95, "loss": 0.05})
run.finish()

# 3. Connect W&B run to FiftyOne
log_wandb_run = foo.get_operator("@harpreetsahota/wandb/log_wandb_run")
log_wandb_run(
    dataset,
    project_name="fiftyone-integration-demo",
    run_name=run.name,
    predictions_field="predictions",  # If you added predictions
    gt_field="ground_truth"            # Ground truth field
)

# 4. View in FiftyOne App
session = fo.launch_app(dataset)

# 5. Open W&B panel to see metrics and visualizations
show_wandb_run = foo.get_operator("@harpreetsahota/wandb/show_wandb_run")
show_wandb_run(dataset)
```

## Features

### ✅ Run Tracking
- Link W&B runs to FiftyOne datasets
- Track which datasets were used for experiments
- Store run metadata (config, metrics, tags, notes)

### ✅ Integrated Visualization
- View W&B dashboard within FiftyOne App
- Side-by-side comparison of data and metrics
- Navigate between projects and runs

### ✅ Field Linking
- Connect prediction fields to specific runs
- Link ground truth fields for evaluation
- Track model versions used for predictions

### ✅ Bidirectional Navigation
- Jump from FiftyOne to W&B dashboard
- See dataset details in W&B (via tags)
- Full URL tracking for easy sharing

## Plugin Operators

The plugin provides the following operators:

1. **`open_wandb_panel`**
   - Opens the W&B panel in FiftyOne App
   - Access via button or operators menu

2. **`log_wandb_run`**
   - Logs a W&B run to your FiftyOne dataset
   - Stores run metadata and links fields

3. **`show_wandb_run`**
   - Shows a specific W&B run in the embedded panel
   - Updates FiftyOne view to show relevant fields

4. **`get_wandb_run_info`**
   - Retrieves detailed information about stored runs
   - Displays config, metrics, and metadata

## Troubleshooting

### "wandb is not installed"

Install W&B:
```bash
pip install wandb
```

### "WANDB_API_KEY not set"

Set your API key:
```bash
export WANDB_API_KEY="your-api-key"
```

Or log in:
```bash
wandb login
```

### "Could not find W&B run"

Make sure:
- The project name is correct
- The run name or ID is correct
- You have access to the project
- The run exists in your W&B account

### "Could not add tags to W&B run"

This is usually a permissions issue. The plugin will continue to work, but tags won't be added to W&B. Check that:
- You have write access to the project
- The run is not in a read-only state

## Advanced Usage

### Working with Multiple Projects

You can track multiple W&B projects in a single FiftyOne dataset:

```python
# Log runs from different projects
log_wandb_run(dataset, project_name="project-1", run_name="run-1")
log_wandb_run(dataset, project_name="project-2", run_name="run-2")

# View specific project
show_wandb_run(dataset)  # Select project from dropdown
```

### Filtering by W&B Metadata

You can use FiftyOne's run system to filter and query:

```python
# Get all W&B runs
runs = [r for r in dataset.list_runs() 
        if dataset.get_run_info(r).config.method == "wandb_run"]

# Get run details
for run in runs:
    info = dataset.get_run_info(run)
    print(f"Run: {info.config.run_name}")
    print(f"Project: {info.config.project}")
    print(f"State: {info.config.state}")
    print(f"URL: {info.config.url}")
```

### Accessing W&B API Directly

The plugin uses the W&B API. You can also use it directly:

```python
import wandb

api = wandb.Api()
runs = api.runs("entity/project")

for run in runs:
    print(f"{run.name}: {run.state}")
    print(f"Config: {dict(run.config)}")
    print(f"Summary: {dict(run.summary)}")
```

## Development

### Building from Source

```bash
# Install dependencies
pip install -r requirements.txt
npm install

# Build TypeScript
npm run build

# Install plugin locally
fiftyone plugins install .
```

### Running in Development Mode

```bash
# Watch for TypeScript changes
npm run dev
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This plugin is released under the Apache 2.0 License.

## Links

- [FiftyOne Documentation](https://docs.voxel51.com/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Plugin GitHub Repository](https://github.com/harpreetsahota204/fiftyone_wandb_plugin)
- [Report Issues](https://github.com/harpreetsahota204/fiftyone_wandb_plugin/issues)

## Support

- [FiftyOne Slack Community](https://slack.voxel51.com/)
- [W&B Community](https://wandb.ai/community)
- [GitHub Issues](https://github.com/harpreetsahota204/fiftyone_wandb_plugin/issues)
