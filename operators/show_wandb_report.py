"""Show W&B Report operator.

This operator displays W&B reports embedded in the FiftyOne App,
allowing users to view custom reports without leaving FiftyOne.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import DEFAULT_WANDB_URL, get_wandb_api


class ShowWandBReport(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="show_wandb_report",
            label="Show W&B Report",
            icon = "/assets/wandb.svg",
            dynamic=True,
            description=(
                "View a W&B report embedded in FiftyOne. "
                "Reports can be embedded unlike the main dashboard."
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Get entity and project from environment or user input
        entity = ctx.secret("FIFTYONE_WANDB_ENTITY")
        project_name = ctx.secret("FIFTYONE_WANDB_PROJECT")
        
        if not entity or not project_name:
            inputs.view(
                "warning",
                types.Warning(
                    label="Configuration Required",
                    description="Please set FIFTYONE_WANDB_ENTITY and FIFTYONE_WANDB_PROJECT "
                               "environment variables to list reports.",
                ),
            )
            # Allow manual input
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
        
        # Fetch available reports
        try:
            api = get_wandb_api(ctx)
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
                description="Choose a W&B report to embed",
                required=True,
                view=types.DropdownView(),
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
        # Get selected report label
        report_label = ctx.params.get("report_label", None)
        report_url = None
        
        if report_label:
            # Fetch reports again to find the URL matching the label
            entity = ctx.secret("FIFTYONE_WANDB_ENTITY")
            project_name = ctx.secret("FIFTYONE_WANDB_PROJECT")
            
            if entity and project_name:
                try:
                    api = get_wandb_api(ctx)
                    reports = list(api.reports(path=f"{entity}/{project_name}", per_page=100))
                    
                    # Find the report that matches the label
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
                except Exception as e:
                    print(f"Error fetching reports: {e}")
        
        # Fallback if no report URL found
        if not report_url:
            entity = ctx.params.get("entity") or ctx.secret("FIFTYONE_WANDB_ENTITY")
            project_name = ctx.params.get("project_name") or ctx.secret("FIFTYONE_WANDB_PROJECT")
            
            if entity and project_name:
                # Open the reports page for the project
                report_url = f"{DEFAULT_WANDB_URL}/{entity}/{project_name}/reports"
            else:
                report_url = DEFAULT_WANDB_URL
        
        # Trigger the panel to embed the report
        ctx.trigger(
            "@harpreetsahota/wandb/embed_report",
            params=dict(url=report_url),
        )
        
        # Open the panel
        ctx.trigger(
            "open_panel",
            params=dict(name="WandBPanel", layout="horizontal", isActive=True),
        )

