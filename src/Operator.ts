import {
  Operator,
  OperatorConfig,
  ExecutionContext,
  registerOperator,
} from "@fiftyone/operators";
import { useSetRecoilState } from "recoil";
import { wandbURLAtom } from "./State";

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
    return {
      setWandBUrl: setWandBUrl,
    };
  }
  async execute({ hooks, params }: ExecutionContext) {
    // Set the URL in state
    hooks.setWandBUrl(params.url);
    
    // Open in new tab immediately
    if (params.url) {
      window.open(params.url, "_blank", "noopener,noreferrer");
    }
  }
}

registerOperator(SetWandBURL, "@harpreetsahota/wandb");
