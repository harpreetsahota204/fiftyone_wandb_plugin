"""Show W&B Report operator.

This operator displays W&B reports embedded in the FiftyOne App,
allowing users to view custom reports without leaving FiftyOne.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ..wandb_helpers import (
    DEFAULT_WANDB_URL,
    get_credentials,
    get_wandb_api,
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
                "View a W&B report embedded in FiftyOne. "
                "Reports can be embedded unlike the main dashboard."
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Get credentials
        entity, api_key, project = get_credentials(ctx)
        
        # Fetch and show projects dropdown
        if entity and api_key and WANDB_AVAILABLE:
            # Get authenticated API (handles login once)
            try:
                api = get_wandb_api(ctx)
            except (ImportError, ValueError) as e:
                inputs.view("error", types.Error(
                    label="Configuration Error",
                    description=str(e)
                ))
                return types.Property(inputs)
            
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
        else:
            inputs.str("project_name", label="W&B Project", required=True)
            return types.Property(inputs)
        
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
            entity, _, project = get_credentials(ctx)
            project_name = project
            
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
            entity, _, project = get_credentials(ctx)
            entity = ctx.params.get("entity") or entity
            project_name = ctx.params.get("project_name") or project
            
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

