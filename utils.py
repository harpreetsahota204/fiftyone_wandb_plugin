"""
Utility functions for W&B plugin.

This file contains helper functions that can be imported and used
independently of the plugin operators.
"""

import wandb


def log_wandb_run_to_fiftyone_dataset(
    sample_collection, project_name, run_id=None, run_name=None
):
    """
    Log a W&B run to a FiftyOne dataset.

    Args:
        sample_collection: The FiftyOne `Dataset` or `DatasetView` 
        project_name: The name of the W&B project
        run_id: The W&B run ID (optional)
        run_name: The W&B run name (optional)
    """
    import fiftyone.operators as foo
    
    # Get the operator and execute it
    log_wandb_run = foo.get_operator("@harpreetsahota/wandb/log_wandb_run")
    return log_wandb_run(
        sample_collection,
        project_name=project_name,
        run_id=run_id,
        run_name=run_name
    )


def show_wandb_run_in_app(dataset):
    """
    Open the W&B panel in FiftyOne App.
    
    Args:
        dataset: The FiftyOne dataset
    """
    import fiftyone.operators as foo
    
    show_wandb_run = foo.get_operator("@harpreetsahota/wandb/show_wandb_run")
    return show_wandb_run(dataset)


def get_wandb_run_info(dataset, run_key):
    """
    Get information about a W&B run stored in FiftyOne.
    
    Args:
        dataset: The FiftyOne dataset
        run_key: The run key in FiftyOne
        
    Returns:
        dict: Run information
    """
    info = dataset.get_run_info(run_key)
    
    return {
        "run_name": info.config.run_name,
        "run_id": info.config.run_id,
        "project": info.config.project,
        "entity": info.config.entity,
        "url": info.config.url,
        "state": info.config.state,
        "config": info.config.config_params,
        "summary": info.config.summary_metrics,
        "tags": info.config.tags,
    }
