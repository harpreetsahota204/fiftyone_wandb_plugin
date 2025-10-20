import {
  Operator,
  OperatorConfig,
  ExecutionContext,
  registerOperator,
} from "@fiftyone/operators";
import { useSetRecoilState } from "recoil";
import { wandbURLAtom, reportModeAtom } from "./State";

class SetWandBURL extends Operator {
  get config(): OperatorConfig {
    return new OperatorConfig({
      name: "set_wandb_url",
      label: "SetWandBURL",
      unlisted: true,
    });
  }
  useHooks(): {} {
    const setWandBUrl = useSetRecoilState(wandbURLAtom);
    const setReportMode = useSetRecoilState(reportModeAtom);
    return {
      setWandBUrl: setWandBUrl,
      setReportMode: setReportMode,
    };
  }
  async execute({ hooks, params }: ExecutionContext) {
    // Set the URL in state
    hooks.setWandBUrl(params.url);
    // Ensure we're in launcher mode (not report mode)
    hooks.setReportMode(false);
    
    // Open in new tab immediately
    if (params.url) {
      window.open(params.url, "_blank", "noopener,noreferrer");
      return { success: true, url: params.url };
    }
    
    return { success: false, error: "No URL provided" };
  }
}

class EmbedReport extends Operator {
  get config(): OperatorConfig {
    return new OperatorConfig({
      name: "embed_report",
      label: "EmbedReport",
      unlisted: true,
    });
  }
  useHooks(): {} {
    const setWandBUrl = useSetRecoilState(wandbURLAtom);
    const setReportMode = useSetRecoilState(reportModeAtom);
    return {
      setWandBUrl: setWandBUrl,
      setReportMode: setReportMode,
    };
  }
  async execute({ hooks, params }: ExecutionContext) {
    if (params.url) {
      // Set the URL in state
      hooks.setWandBUrl(params.url);
      // Set report mode to true (show iframe)
      hooks.setReportMode(true);
      return { success: true, url: params.url, mode: "embed" };
    }
    
    return { success: false, error: "No URL provided" };
  }
}

registerOperator(SetWandBURL, "@harpreetsahota/wandb");
registerOperator(EmbedReport, "@harpreetsahota/wandb");
