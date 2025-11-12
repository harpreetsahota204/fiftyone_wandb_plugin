"""Show W&B Report operator.

This operator displays W&B reports in the FiftyOne App by leveraging
a workaround: W&B allows embedding runs but not reports directly, so we
load a run from the project and provide the report URL for navigation.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import (
    DEFAULT_WANDB_URL,
    get_credentials,
    get_wandb_api,
    prompt_for_missing_credentials,
    WANDB_AVAILABLE,
)


class ShowWandBReport(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="show_wandb_report",
            label="Show W&B Report",
            icon = "/assets/wandb.svg",
            dynamic=True,
            description=(
                "View W&B reports in FiftyOne. Opens a run from the project, "
                "then you can navigate to the report within the W&B interface."
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Check for credentials and prompt if missing - all validation happens here
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
        
        # Get selected project
        project_name = ctx.params.get("project_name")
        if not project_name:
            return types.Property(inputs)
        
        # Fetch available reports (reuse API client from above)
        try:
            reports = list(api.reports(path=f"{entity}/{project_name}", per_page=100))
            
            if len(reports) == 0:
                inputs.view(
                    "warning",
                    types.Warning(
                        label="No Reports Found",
                        description=f"No reports found for {entity}/{project_name}. "
                                   "Create a report in W&B first.",
                    ),
                )
                return types.Property(inputs)
            
            # Create dropdown of reports with display names
            report_choices = types.DropdownView()
            
            for report in reports:
                # Build display label with name and description
                display_name = report.display_name or report.name
                if report.description:
                    # Truncate description if too long
                    desc = report.description[:250] + "..." if len(report.description) > 250 else report.description
                    label = f"{display_name} - {desc}"
                else:
                    label = f"{display_name} - No description provided"
                
                # Use label as the choice value (what user sees)
                report_choices.add_choice(label, label=label)
            
            inputs.enum(
                "report_label",
                report_choices.values(),
                label="Select Report",
                description="Choose a report. We'll load a run from this project, then you can navigate to the report.",
                required=True,
                view=types.DropdownView(),
            )
            
            # Add helpful notice about the workaround
            inputs.view(
                "report_info",
                types.Notice(
                    label="💡 How This Works",
                    description=(
                        "W&B doesn't allow direct report embedding, but we can work around this:\n"
                        "1. We'll load a recent run from this project\n"
                        "2. Once loaded, use W&B's navigation to access your report\n"
                        "3. The report URL will be shown below for easy reference"
                    )
                )
            )
            
        except Exception as e:
            inputs.view(
                "error",
                types.Error(
                    label="Error Loading Reports",
                    description=f"Failed to fetch reports: {str(e)}",
                ),
            )
        
        return types.Property(inputs)

    def execute(self, ctx):
        # Get selected report label and find the report URL
        report_label = ctx.params.get("report_label", None)
        report_url = None
        run_url = None
        
        entity, _, _ = get_credentials(ctx)
        project_name = ctx.params.get("project_name")
        
        if report_label and entity and project_name:
            try:
                api = get_wandb_api(ctx)
                
                # Find the report URL
                reports = list(api.reports(path=f"{entity}/{project_name}", per_page=100))
                for report in reports:
                    display_name = report.display_name or report.name
                    if report.description:
                        desc = report.description[:250] + "..." if len(report.description) > 250 else report.description
                        label = f"{display_name} - {desc}"
                    else:
                        label = f"{display_name} - No description provided"
                    
                    if label == report_label:
                        report_url = report.url
                        break
                
                # Get a recent run from the project to use as entry point
                runs = list(api.runs(path=f"{entity}/{project_name}", per_page=5))
                if runs:
                    run_url = runs[0].url
                else:
                    # No runs available, fallback to project overview
                    run_url = f"{DEFAULT_WANDB_URL}/{entity}/{project_name}"
                    
            except Exception as e:
                print(f"Error fetching data: {e}")
                run_url = f"{DEFAULT_WANDB_URL}/{entity}/{project_name}"
        
        # Fallback URLs if something went wrong
        if not run_url:
            if entity and project_name:
                run_url = f"{DEFAULT_WANDB_URL}/{entity}/{project_name}"
            else:
                run_url = DEFAULT_WANDB_URL
        
        if not report_url:
            if entity and project_name:
                report_url = f"{DEFAULT_WANDB_URL}/{entity}/{project_name}/reports"
            else:
                report_url = DEFAULT_WANDB_URL
        
        # Show the report URL to user for reference
        print(f"📊 Report URL: {report_url}")
        print(f"🚀 Loading via run: {run_url}")
        print(f"💡 Once loaded, navigate to Reports in W&B to access: {report_url}")
        
        # Load the run URL (which W&B allows to embed)
        ctx.trigger(
            "@harpreetsahota/wandb/embed_report",
            params=dict(url=run_url),
        )
        
        # Open the panel
        ctx.trigger(
            "open_panel",
            params=dict(name="WandBPanel", layout="horizontal", isActive=True),
        )

